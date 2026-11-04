"""
LangGraph + Groq + Selenium goal-driven browser agent.

The LLM selects structured browser actions from the current HTML and goal.
Selenium executes only allowlisted actions; arbitrary LLM Python is never exec'd.

Install:
  pip install -r requirements_agentic.txt

.env:
  GROQ_API_KEY=your_new_api_key
  GROQ_MODEL=llama-3.3-70b-versatile
  MAX_STEPS=30
  HEADLESS=false
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
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
WAIT_SECONDS = 12
MAX_HTML_CHARS = 25000
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"


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
    status: str
    final_json: dict[str, Any]
    error: str


class BrowserAction(BaseModel):
    status: Literal["next", "finish"] = "next"
    action: Literal["click", "type", "extract", "select", "scroll", "wait", "none"] = "none"
    locator_by: Optional[Literal["css", "xpath", "id", "name", "class_name", "tag_name", "link_text"]] = None
    locator: Optional[str] = None
    value: Optional[str] = None
    field: Optional[str] = None
    extract_as: Literal["text", "value", "href", "src", "attribute"] = "text"
    attribute: Optional[str] = None
    reason: str = ""
    final_json: dict[str, Any] = Field(default_factory=dict)


BY_MAP = {
    "css": By.CSS_SELECTOR,
    "xpath": By.XPATH,
    "id": By.ID,
    "name": By.NAME,
    "class_name": By.CLASS_NAME,
    "tag_name": By.TAG_NAME,
    "link_text": By.LINK_TEXT,
}


def create_driver():
    options = webdriver.ChromeOptions()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def snapshot(driver):
    WebDriverWait(driver, WAIT_SECONDS).until(
        lambda d: d.find_element(By.TAG_NAME, "body")
    )
    body = driver.find_element(By.TAG_NAME, "body")
    return (
        (body.get_attribute("innerHTML") or "")[:MAX_HTML_CHARS],
        (body.text or "")[:MAX_HTML_CHARS],
    )


def execute_action(driver, action_data):
    """Execute one validated, allowlisted Selenium action."""
    action = BrowserAction.model_validate(action_data)

    if action.action == "none":
        return "No action executed.", {}

    if action.action == "scroll":
        driver.execute_script("window.scrollBy(0, 650);")
        return "Scrolled down.", {}

    if action.action == "wait":
        try:
            seconds = max(1, min(int(action.value or "2"), 10))
        except ValueError:
            seconds = 2
        time.sleep(seconds)
        return f"Waited {seconds} seconds.", {}

    if not action.locator_by or not action.locator:
        raise ValueError("locator_by and locator are required.")

    elements = WebDriverWait(driver, WAIT_SECONDS).until(
        lambda d: d.find_elements(BY_MAP[action.locator_by], action.locator)
    )
    visible = [el for el in elements if el.is_displayed()]
    if not visible:
        raise NoSuchElementException(f"No visible element: {action.locator}")

    if action.action == "extract":
        values = []
        for el in visible:
            if action.extract_as == "value":
                value = el.get_attribute("value")
            elif action.extract_as == "href":
                value = el.get_attribute("href")
            elif action.extract_as == "src":
                value = el.get_attribute("src")
            elif action.extract_as == "attribute":
                if not action.attribute:
                    raise ValueError("attribute is required for extract_as='attribute'")
                value = el.get_attribute(action.attribute)
            else:
                value = el.text.strip()
            if value:
                values.append(value)
        return f"Extracted {len(values)} value(s).", {
            action.field or "extracted_values": values
        }

    el = visible[0]
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)

    if action.action == "click":
        # Form submissions and destructive actions require a separate approval flow.
        tag = (el.tag_name or "").lower()
        input_type = (el.get_attribute("type") or "").lower()
        if tag == "button" or input_type == "submit":
            raise PermissionError("Automatic button/form submission is disabled.")
        el.click()
        time.sleep(1)
        return f"Clicked {action.locator}.", {}

    if action.action == "type":
        el.clear()
        el.send_keys(action.value or "")
        return f"Typed into {action.locator}.", {}

    if action.action == "select":
        Select(el).select_by_visible_text(action.value or "")
        return f"Selected {action.value}.", {}

    raise ValueError(f"Unsupported action: {action.action}")


def build_graph(driver):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GROQ_API_KEY in your .env file.")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key="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",
    )
    structured_llm = llm.with_structured_output(BrowserAction)

    def input_start(state: AgentState):
        driver.get(state["url"])
        html, page_text = snapshot(driver)
        return {
            "html": html,
            "page_text": page_text,
            "current_url": driver.current_url,
            "page_title": driver.title,
            "step_count": 0,
            "history": [],
            "collected_data": {},
            "status": "next",
            "error": "",
        }

    def think(state: AgentState):
        prompt = f"""
You are an agentic browser automation planner.

USER GOAL:
{state['goal']}

CURRENT URL: {state.get('current_url', '')}
PAGE TITLE: {state.get('page_title', '')}

VISIBLE PAGE TEXT:
{state.get('page_text', '')[:10000]}

CURRENT BODY HTML:
{state.get('html', '')[:MAX_HTML_CHARS]}

DATA ALREADY COLLECTED:
{json.dumps(state.get('collected_data', {}), ensure_ascii=False, default=str)}

RECENT ACTION HISTORY:
{json.dumps(state.get('history', [])[-10:], ensure_ascii=False)}

Choose exactly one action using the BrowserAction schema.

Rules:
- status="finish" only when the requested goal is genuinely satisfied.
- Otherwise status="next" and choose one action.
- Use selectors grounded in the supplied HTML; do not invent elements.
- Use click for ordinary navigation links.
- Use type only for non-sensitive text supplied by the user.
- Use extract for visible text, values, hrefs, srcs, or attributes.
- Use scroll/wait when needed for dynamic content.
- Never bypass CAPTCHA, authentication controls, or access restrictions.
- Do not perform payments, purchases, deletion, or irreversible actions.
- Do not submit login/forms automatically. If login is needed, explain in reason
  that the user must complete the login manually.
- Do not claim success just because navigation occurred.
- final_json must contain only information supported by collected data.
"""
        decision = structured_llm.invoke(prompt)
        return {
            "action": decision.model_dump(),
            "status": decision.status,
            "final_json": decision.final_json,
        }

    def run_browser_action(state: AgentState):
        try:
            description, new_data = execute_action(driver, state["action"])
            collected = dict(state.get("collected_data", {}))
            for key, values in new_data.items():
                old = collected.get(key, [])
                if not isinstance(old, list):
                    old = [old]
                if not isinstance(values, list):
                    values = [values]
                collected[key] = list(dict.fromkeys(old + values))

            history = list(state.get("history", []))
            history.append(description)
            html, page_text = snapshot(driver)

            return {
                "html": html,
                "page_text": page_text,
                "current_url": driver.current_url,
                "page_title": driver.title,
                "action_result": description,
                "collected_data": collected,
                "history": history,
                "step_count": state.get("step_count", 0) + 1,
                "error": "",
            }
        except Exception as exc:
            history = list(state.get("history", []))
            history.append(f"Action failed: {type(exc).__name__}: {exc}")
            try:
                html, page_text = snapshot(driver)
            except Exception:
                html, page_text = state.get("html", ""), state.get("page_text", "")
            return {
                "html": html,
                "page_text": page_text,
                "current_url": driver.current_url,
                "page_title": driver.title,
                "action_result": f"Action failed: {exc}",
                "history": history,
                "step_count": state.get("step_count", 0) + 1,
                "error": str(exc),
            }

    def route_after_think(state: AgentState):
        if state.get("status") == "finish":
            return "finish"
        if state.get("step_count", 0) >= MAX_STEPS:
            return "finish"
        return "run"

    graph = StateGraph(AgentState)
    graph.add_node("input", input_start)
    graph.add_node("think", think)
    graph.add_node("run", run_browser_action)

    graph.set_entry_point("input")
    graph.add_edge("input", "think")
    graph.add_conditional_edges(
        "think",
        route_after_think,
        {"run": "run", "finish": END},
    )
    graph.add_edge("run", "think")
    return graph.compile()


def main():
    url = input("Enter URL: ").strip()
    goal = input("Enter your extraction goal: ").strip()

    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    if not goal:
        raise ValueError("Goal cannot be empty.")

    driver = create_driver()
    try:
        app = build_graph(driver)
        initial_state: AgentState = {
            "url": url,
            "goal": goal,
            "collected_data": {},
            "history": [],
            "step_count": 0,
            "status": "next",
            "final_json": {},
        }
        result = app.invoke(initial_state)

        output = {
            "status": result.get("status"),
            "goal": goal,
            "data": result.get("final_json", {}),
            "collected_data": result.get("collected_data", {}),
            "steps": result.get("step_count", 0),
            "history": result.get("history", []),
            "error": result.get("error", ""),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        with open("agent_result.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print("Saved result to agent_result.json")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
