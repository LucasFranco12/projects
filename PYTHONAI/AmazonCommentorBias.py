from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import random
from selenium.common.exceptions import NoSuchElementException

biased_reviewers = 0
unbiased_reviewers = 0

# Function to generate random wait time
def random_wait():
    return random.uniform(2, 5)

# Function to scrape reviews
def scrape_reviews(driver, star_rating, num_pages=20):
    global biased_reviewers, unbiased_reviewers

    star_rating_xpath = f'//*[@id="histogramTable"]/tbody/tr[{star_rating}]/td[1]/a'
    star_rating_link = driver.find_element(By.XPATH, star_rating_xpath)
    star_rating_link.click()
    time.sleep(random_wait())  # Wait for the page to load

    for page in range(num_pages):
        names = []

        review_elements = driver.find_elements(By.CSS_SELECTOR, 'div.review')

        for review in review_elements:
            try:
                # Locate the reviewer name within the review
                name_element = review.find_element(By.CSS_SELECTOR, 'span.a-profile-name')
                name = name_element.text.strip()
                names.append(name)
            except NoSuchElementException:
                print("Name not found for a review.")

        # Scrape additional information from reviewer profiles
        for name in names:
            # Click on reviewer name to go to their profile
            try:
                reviewer_link = driver.find_element(By.XPATH, f'//span[text()="{name}"]/ancestor::a')
                reviewer_link.click()
                time.sleep(random_wait())
            except NoSuchElementException:
                print(f"Could not find link for reviewer: {name}")
                continue

            # Find hearts within the same scope as name_element
            try:
                hearts_element = driver.find_element(By.CSS_SELECTOR, 'span.impact-text')
                hearts = int(hearts_element.text.replace(',', '').strip())
            except NoSuchElementException:
                hearts = 0

            # Check if the reviewer is biased or unbiased
            if hearts > 20:
                unbiased_reviewers += 1
            else:
                biased_reviewers += 1

            # Go back to reviews page
            driver.back()

        # Navigate to the next page
        try:
            next_page_button = driver.find_element(By.XPATH, '//*[@id="cm_cr-pagination_bar"]/ul/li[2]/a')
            next_page_button.click()
            time.sleep(random_wait())  # Wait for the page to load
        except NoSuchElementException:
            print("No next page button found.")
            break

# Main script
url = "https://www.amazon.com/ASUS-Gaming-144HZ-Monitor-VG28UQL1A/dp/B09FP3J623/ref=sr_1_3?crid=TE9Q5MLCMMKT&dib=eyJ2IjoiMSJ9.dhWzeJkbIrHMFd-VxiGyduE8filnFOoK4_GdQAfO8pGzf9zWOOZKqhWw2aO9zS8g_Z9xM6QPJDZjcHvSLkn-xwtl6jwSN0cf9P_nu4rt0BC4M7dneRuwBe8i5GPGJDfogzBb-mK0E76VUpU1aZsISnSDDwPFUy_OZpYuMp8Xntg0e8AmfvRyZuu-a3qvgyBAen2yPAMk2dlHCF-ow7YulnNCJuhY-jm4RGuU1uT7aR0.FqJ4SodN-qqyT_su4a81vA6XaOoe8OiVX--Vs0-v7c8&dib_tag=se&keywords=monitor+144hz+4k&qid=1710469103&sprefix=monitor+144hz+4k%2Caps%2C131&sr=8-3"
driver = webdriver.Chrome()

try:
    # Navigate to the Amazon product page
    driver.get(url)
    time.sleep(random_wait())  # Wait for the page to load

    # Scrape reviews for different star ratings.
    for star in range(1, 6):
        scrape_reviews(driver, star)
        driver.get(url)
        time.sleep(random_wait())  # Wait for the page to load


    print("Biased Reviewers:", biased_reviewers)
    print("Unbiased Reviewers:", unbiased_reviewers)
    #totalRev = biased_reviewrs + unbiased_reviewers

finally:
    # Close the browser window
    driver.quit()
