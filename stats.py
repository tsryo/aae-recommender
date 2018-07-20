import collections
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from aaerec.datasets import Bags
from aminer import unpack_papers, papers_from_files
from fiv import load, parse_en_labels
import re


def compute_stats(A):
    return A.shape[1], A.min(), A.max(), np.median(A, axis=1)[0,0], A.mean(), A.std()


def plot(objects, dataset, title, min_key):
    if min_key != -1:
        objects = {x : objects[x] for x in objects if x >= min_key}
    y_pos = np.arange(len(objects.keys()))
    plt.bar(y_pos, objects.values(), align='center', alpha=0.5)
    plt.xticks(y_pos, objects.keys(), rotation='vertical')
    plt.ylabel('Papers')
    plt.title('Papers by {}'.format(title))
    plt.savefig('papers_by_{}_{}.pdf'.format(title.replace(" ", "_"), dataset))
    # plt.show()
    plt.close()


def paper_by_n_citations(citations):
    '''
    From a dictionary with paper IDs as keys and citation numbers as values
    to a dictionary with citation numbers as keys and paper numbers as values
    '''
    papers_by_citations = {}
    for paper in citations.keys():
        try:
            papers_by_citations[citations[paper]] += 1
        except KeyError:
            papers_by_citations[citations[paper]] = 1

    return papers_by_citations

# path = '../Data/Economics/econbiz62k.tsv'
path = '/data21/lgalke/PMC/citations_pmc.tsv'
dataset = "swp"

if dataset == "dblp" or dataset == "acm" or "swp":
    if dataset != "swp":
        path = '/data22/ivagliano/aminer/'
        path += ("dblp-ref/" if dataset == "dblp" else "acm.txt")
        papers = papers_from_files(path, dataset, n_jobs=1)
    else:
        papers = load("/data22/ivagliano/SWP/FivMetadata.json")

    years, citations = {}, {}
    key = "year" if dataset != "swp" else "date"
    for paper in papers:
        try:
            y = paper[key]
            if key == "date" and len(y) < 4:
                continue
            if key == "date" and len(y) >= 4:
                matches = re.findall(r'.*([1-2][0-9]{3})', y)
                # if no or more than one match skip string
                if len(matches) == 0 or len(matches) > 1:
                    print(paper[key])
                    continue
                else:
                    y = int(matches[0])
                years[y] += 1
            else:
                years[int(paper[key])] += 1
        except KeyError:
            if key not in paper.keys():
                # skip papers without a year
                continue
            years[int(y)] = 0
        except ValueError:
            continue
        if dataset == "dblp":
            try:
                citations[paper["n_citation"]] += 1
            except KeyError:
                citations[paper["n_citation"]] = 1
        elif dataset == "acm":
            if "references" not in paper.keys():
                continue
            for ref in paper["references"]:
                try:
                    citations[ref] += 1
                except KeyError:
                    citations[ref] = 1
        else:
            if "subjects" not in paper.keys():
                continue
            for subject in parse_en_labels(paper["subject"]):
                try:
                    citations[subject] += 1
                except KeyError:
                    citations[subject] = 1

    years = collections.OrderedDict(sorted(years.items()))
    l = list(years.keys())
    print("First year {}, last year {}".format(l[0], l[-1]))
    cnt = 0

    for key, value in years.items():
        cnt += value
        if cnt/len(papers) >= 0.9:
            print("90:10 ratio at year {}".format(key))
            break

    print("Plotting paper distribution by year on file")
    # plot papers from 1970
    plot(years, dataset, "year", 1970)
    if dataset == "acm" or dataset == "swp":
        citations = paper_by_n_citations(citations)

    citations = collections.OrderedDict(sorted(citations.items()))
    x_dim = "citations" if dataset != "swp" else "occurrences"
    print("Plotting paper distribution by number of {} on file".format(x_dim))
    # plot papers with at least 100 citations
    plot(citations, dataset, "number of {}".format(x_dim), 100)

    print("Unpacking {} data...".format(dataset))
    bags_of_papers, ids, side_info = unpack_papers(papers)
    bags = Bags(bags_of_papers, ids, side_info)

else:
    bags = Bags.load_tabcomma_format(path, unique=True)

bags = bags.build_vocab(apply=True)

csr = bags.tocsr()
print("N ratings:", csr.sum())

column_sums = csr.sum(0).flatten()
row_sums = csr.sum(1).flatten()

print(column_sums.shape)
print(row_sums.shape)

FMT = "N={}, Min={}, Max={} Median={}, Mean={}, Std={}"

print("Items per document")
print(FMT.format(*compute_stats(row_sums)))
print("Documents per item")
print(FMT.format(*compute_stats(column_sums)))
