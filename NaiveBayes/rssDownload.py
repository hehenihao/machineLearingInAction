import feedparser


ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sy=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
print(ny,sy)