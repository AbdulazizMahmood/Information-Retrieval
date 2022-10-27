import math
import database as db
import tokenization as tok


def sim(vector1, vector2):
    return dot_product(vector1, vector2)/(length(vector1)*length(vector2))


def dot_product(vector1, vector2):
    sum = 0
    for x, y in zip(vector1, vector2):
        sum += x*y
    return sum


def length(vector):
    sum = 0
    for x in vector:
        sum += x*x
    return math.sqrt(sum)


def create_query_vector(query):
    db.connect_to_database("IR_database")
    terms = db.load_terms()
    post_query = tok.pre_processed_query(query)
    tf = []

    only_terms = []
    for term, _ in terms:
        only_terms.append(term)
    for word, _ in post_query:
        # if word in terms:
            count = 0
            for word2, _ in post_query:
                if word == word2:
                    count += 1
            tf.append((only_terms.index(word), count))
    tf = set(tf)
    query_vector = []
    for _ in terms:
        query_vector.append(0.0)
    for term_id, count in tf:
        df = db.load_df(term_id)
        idf = math.log10(25/df)
        tf_idf = count * idf
        query_vector[term_id] = tf_idf
    return query_vector
