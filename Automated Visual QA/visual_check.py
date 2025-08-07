from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)
driver.get("https://example.com")  # Use any internal/test site here

title = driver.title
if "Example" in title:
    print("Page title check passed.")
else:
    print("Page title check failed!")

screenshot = "screenshot.png"
driver.save_screenshot(screenshot)
print(f"Screenshot saved as {screenshot}")

driver.quit()