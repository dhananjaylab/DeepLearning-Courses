# filename: fetch_nvda_news.py
import requests
from bs4 import BeautifulSoup

# Define the URL for the Google News search about Nvidia news from March 25 to April 22, 2024
url = "https://www.google.com/search?q=Nvidia+news+March+2024+to+April+2024&tbm=nws&tbs=cdr:1,cd_min:3/25/2024,cd_max:4/22/2024"

# Make the request to the website
response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all news items, each news item is contained in a 'g-card' class
news_items = soup.find_all('div', class_='g-card')

# Display the title and link of each news article
for news_item in news_items:
    title = news_item.find('div', class_='JheGif nDgy9d').text
    link = news_item.find('a')['href']
    print(f"Title: {title}\nLink: {link}\n")

# Check if results were found
if not news_items:
    print("No news articles found. Consider adjusting your search or checking other sources.")