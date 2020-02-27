import json
import math
import numpy as np
import sys
from annoy import AnnoyIndex


def loadQueries(filename):
    queries = set()
    with open(filename,'r') as f:
        for l in f:
            queries.add(l.strip())
    return queries

def loadVectors(filename, queries):
    i = 0
    triples = [{},{},[]] #Query2Idx, idX2Query, list of vectors
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            query = l[0]
            vectors = l[1].split(' ')
            if query in queries:
                triples[0][query] = j
                triples[1][j] = query
                triples[2].append(np.array(vectors,dtype=float))
                i += 1
    return triples

def generateAnnoy(triples, annoyFilename, dimensions=100, treeSize = 100):
    idx2vec = np.array(artificial[2])
    ann = AnnoyIndex(dimensions)
    for j in range(len(artificial[2])):
        ann.add_item(j,idx2vec[j])
    print('Done Adding items to AnnoyIndex')
    ann.build(treesize)
    print('Done Building AnnoyIndex')
    ann.save(annoyFilename)
    print(ann.get_nns_by_item(0, 1000))
    v = ann.get_nns_by_item(0, 10, search_k=-1, include_distances=False)[0]
    ann.get_nns_by_vector(v, 10, search_k=-1, include_distances=False)
    ann.get_distance(i, j)
    return ann

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: generate_query_ann.py <queries> <vectors> <annFilename> ")
        exit(-1)
    else:
        print("Loading Queries")
        queries = loadQueries(sys.argv[1])
        #Run regular embeddings
        print("Loading Query Vectors")
        triples = loadVectors(sys.argv[2], queries)
        print("Building Annnoy Query Embeddings")
        annoyEmbedding = generateAnnoy(triples, sys.argv[3])
