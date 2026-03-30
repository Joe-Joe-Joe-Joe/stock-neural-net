import logging
logging.getLogger('newspaper').setLevel(logging.ERROR)
from newspaper.google_news import GoogleNewsSource
from newspaper import ArticleException
from newspaper.article import ArticleDownloadState
from datetime import datetime, timedelta
import json

OUTPUT_DIR = 'scraped_news'

company_names=["Tesla",
               "McDonald's",
               "Meta"]

# organized in order of left to center to right leaning news sources
# according to https://www.allsides.com/media-bias/media-bias-chart as of 2024-06-01
site_names=["theguardian.com",
            "cnn.com", 
            "bbc.com", 
            "npr.org",
            "marketwatch.com",
            "foxbusiness.com",
            "nypost.com",
            "aljazeera.com", 
            "cbc.ca"]
desired_keys=["url", 
              "title", 
              "text", 
              "authors", 
              "publish_date", 
              "meta_site_name"]

start_date = datetime(2025,5,28)
end_date = datetime(2025,12,31)

while start_date<end_date:
    source = GoogleNewsSource(
        language='en',
        country='US',
        start_date=start_date,
        end_date=start_date+timedelta(days=6),
        max_results=1,
        number_threads=5
    )

    for company_name in company_names:
        broken_flag=False
        storage={start_date.strftime("%Y_%m_%d"): []}
        for site_name in site_names:
            source.build(top_news=False,keyword=company_name+" site:"+site_name)
            try:
                source.articles[0].download()
                if source.articles[0].download_state != ArticleDownloadState.SUCCESS:
                    print(f"Error downloading from {site_name}, skipping article...")
                    continue
                source.articles[0].parse()
                article_data = source.articles[0].to_json(as_string=False)
                keys_copy = list(article_data.keys())
                for key in keys_copy:
                    if key not in desired_keys:
                        del article_data[key]
                article_data["company"] = company_name
                storage[start_date.strftime("%Y_%m_%d")].append(article_data)
            except ArticleException as e:
                print(f"Article error from {site_name}: {e}")
            except Exception as e:
                print(f"Error processing article from {site_name}: {e}")
                broken_flag=True
                break

        if not broken_flag:
            with open(f'{OUTPUT_DIR}/news_{company_name}_{start_date.strftime("%Y_%m_%d")}.json', 'w') as f:
                json.dump(storage, f, indent=4)
                print(f"Exported data for {company_name} on {start_date.strftime('%Y_%m_%d')}")

    start_date+=timedelta(days=7)