from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

# fro img to text
import base64
from io import BytesIO
from PIL import Image
import pytesseract
import time


def login():
    driver = webdriver.Chrome()
    driver.get("https://skillgun.com/StudentLogin.aspx")
    #driver.maximize_window()

    inputs = driver.find_elements(By.TAG_NAME, "input")
    print("Total inputs:", len(inputs))

    for i, element in enumerate(inputs):
        if element.get_attribute("placeholder")=='Phone Number':
            element.send_keys("8639569697")

    for i, element in enumerate(inputs):
        if element.get_attribute("placeholder")=='Email address':
            element.send_keys("suryavardhanc8@gmail.com")
        elif element.get_attribute("placeholder")=='Password':
            element.send_keys("vardhan@321")
        elif element.get_attribute('type')=="checkbox" and element.is_displayed():
            element.click()

    images = driver.find_elements(By.TAG_NAME, "img")
    for i, element in enumerate(images):
        return data_extract(driver,element.get_attribute("src"),inputs)

def data_extract(driver,data,inputs):
    # Extract Base64 portion
    base64_data = data.split(",", 1)[1]

    # Decode
    image_bytes = base64.b64decode(base64_data)

    # Convert directly to PIL image
    image = Image.open(BytesIO(image_bytes))

    #print("Image size:", image.size)

    # OCR
    text = pytesseract.image_to_string(image)

    #print("Content:")

    l=[int(i) for i in text.split(" ")]
    #print(l)
    if len(l)==4:
        return final_submit(driver,l,inputs)
    else:
        return login()



def final_submit(driver,l,inputs):
    result1=""
    l.sort()
    for i, element in enumerate(inputs):
        if 'biggest' in element.get_attribute("placeholder"):
            if 'second' in element.get_attribute("placeholder"):
                element.send_keys(str(l[2]))
            else:
                element.send_keys(str(l[3]))
            
        if 'smallest' in element.get_attribute("placeholder"):
            if 'second' in element.get_attribute("placeholder"):
                element.send_keys(str(l[1]))
            else:
                element.send_keys(str(l[0]))

        if element.get_attribute('type')=="submit" and element.is_displayed():
            element.click()
            break
    #input("hi :")

    
    a = driver.find_elements(By.TAG_NAME, "a")

    for link in a:
        
        # Safely fetch the core text and URL attributes
        link_text = link.text.strip().lower()
        #print(link_text)
        result1=link_text
        if "new placement drive" in link_text:
            link.click()
            break
    
    #print("**"*10,"fileld","**"*10)
    field = driver.find_elements(By.TAG_NAME, "fieldset")

    # Assuming you already ran: field = driver.find_elements(By.TAG_NAME, "fieldset")

    for fieldset in field:
        # 1. Get the visible text
        #print("Text content:", fieldset.text)
        result2=fieldset.text
        # 2. Get the ID or Class name
        fieldset_id = fieldset.get_attribute("id")
        fieldset_class = fieldset.get_attribute("class")
        #print(f"ID: {fieldset_id} | Class: {fieldset_class}")
        
        # 3. Check if it's visible to the user
        if fieldset.is_displayed():
            #print("This fieldset is visible.")
            pass
        #print("-" * 20)  # Separator line

    a = driver.find_elements(By.TAG_NAME, "a")
    final_dict={}
    j=1
    

    for i in range(len(a)):
        #link=driver.find_elements(By.TAG_NAME, "a")
        link=a[i]
        link_text = link.text.strip().lower()
        if "show details" in link_text:
            #print("link 1","--"*20)
            final_dict[j]=fetch_individual(link,driver)
            #print(j,final_dict[j])
            j+=1
    driver.quit()
    return final_dict


def fetch_individual(link,driver):
    link.click()
    elements = driver.find_elements(By.XPATH, "//input | //span")
    res=[]
    for element in elements:
        tag_type = element.tag_name          
        if not element.is_displayed():
            continue            
        if tag_type == "input":
            input_value = element.get_attribute("value")          
            res.append(['input',input_value])                
        elif tag_type == "span":
            span_text = element.text.strip()
            if span_text:
                res.append(['span',span_text])
    driver.back()
    return collect_details(res)
def collect_details(lis):
    flag=0
    next=0
    dic={}
    for i in range(len(lis)):
        if lis[i][0]=='span' and 'Company Name' in lis[i][1]:
            #print("Company Name :")
            flag=1
            next=1
        elif lis[i][0]=='input' and next==1:
            #print(lis[i][1])
            if lis[i][1]!="":
                dic[lis[i-1][1]]=lis[i][1]
            #print("--"*10)
            next=0
        elif lis[i][0]=='span' and flag==1 and lis[i][0]!='Technology :':
            #print(lis[i][1])
            next=1
    return dic

    """
    inputs = driver.find_elements(By.TAG_NAME, "input")
    for i, element in enumerate(inputs):
        print("input-",i,element.get_attribute("value"))

    spans = driver.find_elements(By.TAG_NAME, "span")
    for i,span in enumerate(spans):
        span_text = span.text.strip()
        print("span-",i,span_text)"""

    #driver.quit()




    #return result2

print("Number jobs available:",login())