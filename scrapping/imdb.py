from bs4 import BeautifulSoup
import requests

list_of_directors = []
list_of_actors = []
stars = []

html = requests.get("https://www.imdb.com/search/title/?genres=sci_fi&sort=user_rating,desc&title_type=feature&num_votes=25000,&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=5aab685f-35eb-40f3-95f7-c53f09d542c3&pf_rd_r=E40TDJEMENX3WJ81KT4Q&pf_rd_s=right-6&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_gnr_17")
soup = BeautifulSoup(html.content, 'html.parser')

ratings = soup.find_all('div',class_="inline-block ratings-imdb-rating")

for rating in ratings:

    stars.append(rating.text.replace("\n",""))
    
print(stars)


maindiv = soup.find_all('div',class_="lister-item-content")

for movie in maindiv:
    names = maindiv[0].find_all('p')[2].text
    names = names.split("|")
    list_of_directors.append(names[0])
    list_of_actors.append(names[ len(names) -1 ])




html = requests.get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=sci-fi&sort=user_rating,desc&start=51&ref_=adv_nxt")

for rating in ratings:
    
    stars.append(rating.text.replace("\n",""))
    
print(stars)

maindiv = soup.find_all('div',class_="lister-item-content")
for movie in maindiv:
    names = maindiv[0].find_all('p')[2].text
    names = names.split("|")
    list_of_directors.append(names[0])
    list_of_actors.append(names[ len(names) -1 ])
 
print(list_of_actors)
print(list_of_directors)
