import feedparser
import pandas as pd
from datetime import datetime
import os


rss_url = "https://news.google.com/rss/search?q=food+industry"


news = feedparser.parse(rss_url)


new_articles = []


for article in news.entries[:10]:

    new_articles.append({
        "Date": datetime.now(),
        "Title": article.title,
        "Link": article.link
    })


new_df = pd.DataFrame(new_articles)


file_path = "data/news_data.csv"


if os.path.exists(file_path):

    old_df = pd.read_csv(file_path)

    combined_df = pd.concat(
        [old_df, new_df]
    )

    combined_df = combined_df.drop_duplicates(
        subset=["Title"]
    )

else:

    combined_df = new_df


combined_df.to_csv(
    file_path,
    index=False
)


print("Database Updated Successfully")
print(
    "Total News:",
    len(combined_df)
)
