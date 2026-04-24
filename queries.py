from pyspark import SparkContext

sc = SparkContext(appName="Project2")

# Load the nodes and edges data into RDDs
nodes_rdd = sc.textFile("data/nodes.tsv")
edges_rdd = sc.textFile("data/edges.tsv")

# Remove header lines
nodes_rdd = nodes_rdd.filter(lambda line: not line.startswith("id"))
edges_rdd = edges_rdd.filter(lambda line: not line.startswith("source"))

# Split the lines into columns
nodes = nodes_rdd.map(lambda line: line.split("\t"))
edges = edges_rdd.map(lambda line: line.split("\t"))


# Filters out everything that's not a drug and keep only the drug id and name as pair
drug = (
    nodes.filter(lambda parts: parts[2] == "Compound")
    .map(lambda parts: (parts[0], parts[1])) 
)
# Query 1
def query1(edges_rdd):
    # Filters edges for gene and disease relationships separately
    gene_edges = {"CbG", "CdG", "CuG"}
    disease_edges = {"CtD", "CpD"}
    gene = edges_rdd.filter(lambda parts: parts[1] in gene_edges)
    disease = edges_rdd.filter(lambda parts: parts[1] in disease_edges)

    # Count number of genes per drug, each (drug, gene) pair is only counted once
    gene_count = gene.map(lambda parts: (parts[0], parts[2])).distinct()\
        .map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
    # Count number of diseases per drug, each (drug, disease) pair is only counted once
    disease_count = disease.map(lambda parts: (parts[0], parts[2])).distinct()\
        .map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

    def join_counts(x):
        drug_id = x[0]
        g_count = x[1][0]
        d_count = x[1][1]

        # For edge cases where a drug has no associated diseases, we set the count to 0
        if d_count is None:
            d_count = 0
        return (drug_id, (g_count, d_count))

    # Combine/Join the drug RDD with the gene and disease counts
    joined = gene_count.leftOuterJoin(disease_count).map(join_counts)

    query1_result = joined.sortBy(lambda x: x[1][0], ascending=False).take(5)
    print("Query 1 - Top 5 drugs sorted by number of genes:")
    for drug_id, (gc, dc) in query1_result:
        print(f"{drug_id} | Genes: {gc} | Diseases: {dc}")
    print("\n")
    return joined

def query2(edges_rdd):
    disease_edges = {"CtD", "CpD"}
    # Filter edges for disease relationships
    disease = edges_rdd.filter(lambda parts: parts[1] in disease_edges)

    # For each disease, count the number of drugs that treat it
    drug_count = disease.map(lambda parts: (parts[2], 1)).reduceByKey(lambda a, b: a + b)
    
    # count how many diseases have the same drug_count
    disease_drug_count = drug_count.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)
    query2_result = disease_drug_count.sortBy(lambda x: x[1], ascending=False).take(5)
    print("Query 2 - Top 5 diseases and their associated drug count:")
    for i, count in query2_result:
        print(f"{i} drugs -> {count} diseases")
    print("\n")

def query3(joined):
    # Reuse aggregated results from Query 1 to avoid recomputing the gene and disease counts
    # Sort by gene count and get the top 5 drug names
    query3_result = joined.sortBy(lambda x: x[1][0], ascending=False).take(5)
    print("Query 3 - Top 5 drug names sorted by number of genes:")
    for drug_id, (gc, dc) in query3_result:
        drug_name = drug.lookup(drug_id)[0]  # lookup the drug name using the drug RDD
        print(f"{drug_name} -> {gc}")

# Run all queries
joined = query1(edges)
query2(edges)
query3(joined)

# Exit the SparkContext
sc.stop()