from pathlib import Path

code = r"""
Agentic web extraction using LangGraph + ChatGroq + Selenium.

Safety model:
- The LLM returns structured browser actions, never executable Python.
- Selenium executes only a small allowlist of actions.
- CAPTCHA / human verification is handed to the user; this script does not solve or bypass it.
- Credentials are read from environment variables, never hardcoded.
"""

import json
import os
import time
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "25"))
MAX_HTML_CHARS = int(os.getenv("MAX_HTML_CHARS", "30000"))
WAIT_SECONDS = int(os.getenv("SELENIUM_WAIT_SECONDS", "12"))
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
ALLOW_FORM_SUBMIT = os.getenv("ALLOW_FORM_SUBMIT", "false").lower() == "true"

# Only these environment variables can be used for secret form values.
SECRET_ENV_KEYS = ("LOGIN_EMAIL", "LOGIN_PASSWORD", "LOGIN_PHONE")


# ---------------------------------------------------------------------------
# LLM structured outputs
# ---------------------------------------------------------------------------

class BrowserAction(BaseModel):
    action: Literal[
        "click",
        "type",
        "type_secret",
        "select",
        "extract",
        "scroll",
        "wait",
        "human",
        "finish",
    ] = Field(description="One safe browser action.")
    locator_by: Optional[
        Literal[
            "css",
            "xpath",
            "id",
            "name",
            "class_name",
            "tag_name",
            "link_text",
            "partial_link_text",
        ]
    ] = None
    locator: Optional[str] = None
    value: Optional[str] = Field(
        default=None,
        description="Text to type, select, wait duration, or attribute name."
    )
    secret_key: Optional[Literal[
        "LOGIN_EMAIL", "LOGIN_PASSWORD", "LOGIN_PHONE"
    ]] = None
    field: Optional[str] = Field(
        default=None,
        description="JSON key to use when action is extract."
    )
    extract_as: Literal["text", "value", "href", "src", "attribute"] = "text"
    attribute: Optional[str] = None
    reason: str = ""


class GoalAssessment(BaseModel):
    complete: bool = Field(
        description="True only when all information requested by the user is present."
    )
    missing_items: list[str] = Field(default_factory=list)
    final_json: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    url: str
    goal: str
    html: str
    page_text: str
    current_url: str
    page_title: str
    action: dict[str, Any]
    action_result: str
    collected_data: dict[str, Any]
    history: list[str]
    step_count: int
    complete: bool
    final_json: dict[str, Any]
    missing_items: list[str]
    error: str


# ---------------------------------------------------------------------------
# Selenium helpers
# ---------------------------------------------------------------------------

def build_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def locator_tuple(by: str, value: str):
    mapping = {
        "css": By.CSS_SELECTOR,
        "xpath": By.XPATH,
        "id": By.ID,
        "name": By.NAME,
        "class_name": By.CLASS_NAME,
        "tag_name": By.TAG_NAME,
        "link_text": By.LINK_TEXT,
        "partial_link_text": By.PARTIAL_LINK_TEXT,
    }
    if by not in mapping:
        raise ValueError(f"Unsupported locator strategy: {by}")
    return mapping[by], value


def looks_like_human_challenge(driver: webdriver.Chrome, html: str) -> bool:
    """Detect common challenge markers and ask the user to complete them manually."""
    lowered = html.lower()
    markers = (
        "g-recaptcha",
        "hcaptcha",
        "cf-turnstile",
        "captcha",
        "verify you are human",
        "security check",
    )
    if any(marker in lowered for marker in markers):
        return True

    try:
        frames = driver.find_elements(By.TAG_NAME, "iframe")
        for frame in frames:
            title = (frame.get_attribute("title") or "").lower()
            name = (frame.get_attribute("name") or "").lower()
            if any(word in title + " " + name for word in ("captcha", "challenge")):
                return True
    except WebDriverException:
        pass

    return False


def collect_page_snapshot(driver: webdriver.Chrome) -> tuple[str, str]:
    WebDriverWait(driver, WAIT_SECONDS).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    body = driver.find_element(By.TAG_NAME, "body")
    html = body.get_attribute("outerHTML") or ""
    visible_text = body.text or ""
    return html[:MAX_HTML_CHARS], visible_text[:MAX_HTML_CHARS]


def execute_browser_action(
    driver: webdriver.Chrome,
    action_data: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Execute one validated action. Returns (description, extracted values)."""
    action = BrowserAction.model_validate(action_data)
    extracted: dict[str, Any] = {}

    if action.action == "finish":
        return "Planner requested verification of the current results.", extracted

    if action.action == "human":
        print("\n[HUMAN ACTION REQUIRED]")
        print(action.reason or "Please complete the required step in the browser.")
        input("After completing it in the browser, press Enter here...")
        return "User completed the manual browser step.", extracted

    if action.action == "wait":
        seconds = 2
        if action.value:
            try:
                seconds = max(1, min(int(action.value), 15))
            except ValueError:
                pass
        time.sleep(seconds)
        return f"Waited {seconds} seconds.", extracted

    if action.action == "scroll":
        driver.execute_script("window.scrollBy(0, 700);")
        return "Scrolled down the page.", extracted

    if not action.locator_by or not action.locator:
        raise ValueError("This action requires locator_by and locator.")

    by, locator = locator_tuple(action.locator_by, action.locator)
    elements = WebDriverWait(driver, WAIT_SECONDS).until(
        lambda d: d.find_elements(by, locator)
    )
    elements = [el for el in elements if el.is_displayed()]

    if not elements:
        raise NoSuchElementException(f"No visible element found: {action.locator}")

    if action.action == "extract":
        values = []
        for element in elements[:100]:
            if action.extract_as == "value":
                item = element.get_attribute("value")
            elif action.extract_as == "href":
                item = element.get_attribute("href")
            elif action.extract_as == "src":
                item = element.get_attribute("src")
            elif action.extract_as == "attribute":
                if not action.attribute:
                    raise ValueError("Set attribute when extract_as='attribute'.")
                item = element.get_attribute(action.attribute)
            else:
                item = element.text.strip()

            if item:
                values.append(item)

        key = action.field or "extracted_values"
        extracted[key] = values
        return f"Extracted {len(values)} value(s) into '{key}'.", extracted

    element = elements[0]

    if action.action == "click":
        tag = (element.tag_name or "").lower()
        input_type = (element.get_attribute("type") or "").lower()
        if (tag == "button" or input_type == "submit") and not ALLOW_FORM_SUBMIT:
            raise PermissionError(
                "Form/button submission is disabled. Review the target site, then "
                "set ALLOW_FORM_SUBMIT=true in .env if submission is intended."
            )
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});", element
        )
        WebDriverWait(driver, WAIT_SECONDS).until(
            lambda d: element.is_displayed() and element.is_enabled()
        )
        element.click()
        time.sleep(1)
        return f"Clicked element located by {action.locator_by}={locator}.", extracted

    if action.action == "type":
        element.clear()
        element.send_keys(action.value or "")
        return f"Typed text into {action.locator_by}={locator}.", extracted

    if action.action == "type_secret":
        if action.secret_key not in SECRET_ENV_KEYS:
            raise ValueError("The requested secret key is not allowed.")
        secret = os.getenv(action.secret_key or "")
        if not secret:
            raise ValueError(
                f"{action.secret_key} is not set in your environment/.env file."
            )
        element.clear()
        element.send_keys(secret)
        return f"Entered value from environment variable {action.secret_key}.", extracted

    if action.action == "select":
        Select(element).select_by_visible_text(action.value or "")
        return f"Selected '{action.value}' from dropdown.", extracted

    raise ValueError(f"Unsupported action: {action.action}")


# ---------------------------------------------------------------------------
# Build the agent graph
# ---------------------------------------------------------------------------

def build_agent(driver: webdriver.Chrome):
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    planner_llm = llm.with_structured_output(BrowserAction)
    verifier_llm = llm.with_structured_output(GoalAssessment)

    session = {"opened": False}

    def collect_html(state: AgentState) -> dict[str, Any]:
        if not session["opened"]:
            driver.get(state["url"])
            session["opened"] = True

        html, page_text = collect_page_snapshot(driver)

        if looks_like_human_challenge(driver, html):
            print(
                "\nA CAPTCHA or human-verification challenge was detected. "
                "Complete it manually in the browser."
            )
            input("Press Enter after completing the challenge...")
            html, page_text = collect_page_snapshot(driver)

        return {
            "html": html,
            "page_text": page_text,
            "current_url": driver.current_url,
            "page_title": driver.title,
        }

    def plan_action(state: AgentState) -> dict[str, Any]:
        prompt = f"""
You are a browser navigation and data-extraction planner.

USER GOAL:
{state['goal']}

CURRENT PAGE:
URL: {state.get('current_url', '')}
TITLE: {state.get('page_title', '')}

VISIBLE PAGE TEXT:
{state.get('page_text', '')[:12000]}

BODY HTML:
{state.get('html', '')[:MAX_HTML_CHARS]}

ALREADY COLLECTED DATA:
{json.dumps(state.get('collected_data', {}), ensure_ascii=False, default=str)}

RECENT ACTION HISTORY:
{chr(10).join(state.get('history', [])[-8:])}

ALLOWED SECRET ENVIRONMENT VARIABLE NAMES:
{", ".join(SECRET_ENV_KEYS)}

Choose exactly ONE next action:
- click: click a visible link or non-submit control to navigate toward the goal.
- type: type non-sensitive text into a field.
- type_secret: fill a sensitive field using one of the allowed environment variable names.
- select: choose a visible dropdown option.
- extract: collect text/value/href/src/attribute from matching visible elements.
- scroll: scroll down to reveal more page content.
- wait: wait briefly for dynamic content.
- human: ask the user to complete a legitimate manual step.
- finish: request verification if you believe the goal is satisfied.

Rules:
1. Use a locator that is actually supported by the supplied HTML.
2. Prefer id, name, stable CSS selectors, or XPath based on visible attributes.
3. Do not invent selectors or page content.
4. Do not generate or execute Python/JavaScript code.
5. Never attempt to solve, decode, or bypass CAPTCHA or access controls. Use human action.
6. Extract only information relevant to the user's goal.
7. For extraction, set field to a meaningful JSON key.
8. For a password/email/phone field, use type_secret and the matching environment variable.
9. If there is not enough information, scroll, wait, or navigate to the most relevant link.
10. Do not click purchase, payment, delete, or other irreversible controls.
"""
        plan = planner_llm.invoke(prompt)
        return {"action": plan.model_dump()}

    def execute_action(state: AgentState) -> dict[str, Any]:
        try:
            description, new_data = execute_browser_action(
                driver, state.get("action", {})
            )
            combined = dict(state.get("collected_data", {}))
            for key, value in new_data.items():
                # Keep earlier results if the same field is extracted again.
                previous = combined.get(key, [])
                if not isinstance(previous, list):
                    previous = [previous]
                incoming = value if isinstance(value, list) else [value]
                combined[key] = list(dict.fromkeys(previous + incoming))

            history = list(state.get("history", []))
            history.append(description)

            return {
                "action_result": description,
                "collected_data": combined,
                "history": history,
                "step_count": state.get("step_count", 0) + 1,
                "error": "",
            }
        except (
            Exception
        ) as exc:
            history = list(state.get("history", []))
            history.append(f"Action failed: {type(exc).__name__}: {exc}")
            return {
                "action_result": f"Action failed: {type(exc).__name__}: {exc}",
                "history": history,
                "step_count": state.get("step_count", 0) + 1,
                "error": str(exc),
            }

    def verify_goal(state: AgentState) -> dict[str, Any]:
        if state.get("step_count", 0) >= MAX_STEPS:
            return {
                "complete": False,
                "missing_items": ["Maximum agent step count reached."],
                "final_json": dict(state.get("collected_data", {})),
                "error": "Maximum agent step count reached.",
            }

        prompt = f"""
You are a strict data-extraction verifier.

USER GOAL:
{state['goal']}

CURRENT PAGE VISIBLE TEXT:
{state.get('page_text', '')[:10000]}

DATA ACTUALLY EXTRACTED BY SELENIUM:
{json.dumps(state.get('collected_data', {}), ensure_ascii=False, default=str)}

LAST ACTION RESULT:
{state.get('action_result', '')}

Determine whether the requested goal is fully satisfied.

Rules:
- Do not invent, infer, or fabricate values.
- final_json must contain only values present in the extracted data above.
- If the goal asks for multiple records, verify that the required records/fields are present.
- If information is missing, set complete=false and list the missing items.
- If complete=true, return a clean JSON object containing the requested information.
- If the last action failed, normally set complete=false.
"""
        assessment = verifier_llm.invoke(prompt)
        return {
            "complete": assessment.complete,
            "missing_items": assessment.missing_items,
            "final_json": assessment.final_json,
        }

    def route_after_verification(state: AgentState) -> str:
        if state.get("complete"):
            return "finish"
        if state.get("step_count", 0) >= MAX_STEPS:
            return "finish"
        return "collect_html"

    graph = StateGraph(AgentState)
    graph.add_node("collect_html", collect_html)
    graph.add_node("plan_action", plan_action)
    graph.add_node("execute_action", execute_action)
    graph.add_node("verify_goal", verify_goal)

    graph.set_entry_point("collect_html")
    graph.add_edge("collect_html", "plan_action")
    graph.add_edge("plan_action", "execute_action")
    graph.add_edge("execute_action", "verify_goal")
    graph.add_conditional_edges(
        "verify_goal",
        route_after_verification,
        {
            "collect_html": "collect_html",
            "finish": END,
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    url = input("Enter the starting URL: ").strip()
    goal = input(
        "Describe exactly what information you want to extract: "
    ).strip()

    if not url.startswith(("http://", "https://")):
        raise ValueError("Please provide a valid http:// or https:// URL.")
    if not goal:
        raise ValueError("The extraction goal cannot be empty.")

    driver = build_driver()
    agent = build_agent(driver)

    initial_state: AgentState = {
        "url": url,
        "goal": goal,
        "collected_data": {},
        "history": [],
        "step_count": 0,
        "complete": False,
        "final_json": {},
        "missing_items": [],
        "error": "",
    }

    try:
        result = agent.invoke(initial_state)

        print("\n========== AGENT RESULT ==========")
        print(json.dumps(
            {
                "complete": result.get("complete", False),
                "missing_items": result.get("missing_items", []),
                "data": result.get("final_json", {}),
                "steps": result.get("step_count", 0),
                "history": result.get("history", []),
                "error": result.get("error", ""),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ))

        output_file = "extracted_result.json"
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(
                result.get("final_json", {}),
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(f"\nSaved result to: {output_file}")

    finally:
        input("\nPress Enter to close the browser...")
        driver.quit()


if __name__ == "__main__":
    main()
