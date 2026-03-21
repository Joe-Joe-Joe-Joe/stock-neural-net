import logging
logging.getLogger('newspaper').setLevel(logging.WARNING)
from newspaper.google_news import GoogleNewsSource
from datetime import datetime, timedelta
import json

company_names=["Tesla","McDonald's","Meta"]
site_names=["cnn.com", "bbc.com", "aljazeera.com", "cbc.ca", "theguardian.com"]
desired_keys=["url", "title", "text", "authors", "publish_date", "meta_site_name"]

end_date=datetime(2025,1,8)
while end_date<datetime(2026,1,7):
    start_date=end_date-timedelta(days=7)
    source = GoogleNewsSource(
        language='en',
        country='US',
        start_date=start_date,
        end_date=end_date,
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
                source.articles[0].parse()
                article_data = source.articles[0].to_json(as_string=False)
                keys_copy = list(article_data.keys())
                for key in keys_copy:
                    if key not in desired_keys:
                        del article_data[key]
                article_data["company"] = company_name
                storage[start_date.strftime("%Y_%m_%d")].append(article_data)
            except Exception as e:
                print(f"Error processing article from {site_name}: {e}")
                broken_flag=True
                break

        if not broken_flag:
            with open('scraped_news/news_'+company_name+"_"+start_date.strftime("%Y_%m_%d")+'.json', 'w') as f:
                json.dump(storage, f, indent=4)

    end_date+=timedelta(days=7)