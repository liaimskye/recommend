import spacy
nlp = spacy.load("en_core_web_md")

with open("movies.txt", "r") as file:
    descriptions = file.readlines()


movie_descrip = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."
model_description = nlp(movie_descrip)


def recomendation(description, database):


    highest_sim_num = 0
    highest_sim_descrip = None

    for sentance in database:
        sentance_chopped = sentance[9:]
        sim_score = nlp(sentance_chopped).similarity(model_description)
        if sim_score > highest_sim_num:
            highest_sim_descrip = sentance
            highest_sim_num = sim_score


    return highest_sim_descrip,highest_sim_num


movie_recomendation,similarity_score = recomendation(movie_descrip, descriptions)

print(movie_recomendation,"\n",similarity_score)
    
    

