import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
import sys
import os
import pandas as pd

from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def cluster_banks(
    data: pd.DataFrame,
    bank_id_column: str,
    n_clusters: int = None,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    
    # Extract features and scale them
    data = data.drop(columns=["Timestamp", "Is Laundering"])
    X = data.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
   
    print("Starting clustering")
    # Perform final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create temporary DataFrame with cluster assignments
    temp_df = pd.DataFrame({
        'bank_id': data[bank_id_column],
        'cluster': cluster_labels
    })
    
    # Find most common cluster for each bank
    bank_cluster_modes = {}
    for bank in temp_df['bank_id'].unique():
        bank_clusters = temp_df[temp_df['bank_id'] == bank]['cluster']
        most_common_cluster = Counter(bank_clusters).most_common(1)[0][0]
        bank_cluster_modes[bank] = most_common_cluster
    
    print("done with most common")
    # Group banks by their most common cluster
    clusters = defaultdict(list)
    for bank, cluster in bank_cluster_modes.items():
        clusters[cluster].append(bank)
    
    return dict(clusters)

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

raw = dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,FromBank, ToBank, Is Laundering\n"

firstTs = -1

with open(outPath, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):
        datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

        fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

        fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"To Bank"] + raw[i,4]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(raw[i,"Amount Received"])
        amountPaidOrig = float(raw[i,"Amount Paid"])

        fromBank = raw[i,"From Bank"]
        toBank = raw[i,"To Bank"]

        isl = int(raw[i,"Is Laundering"])

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%s,%s,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,fromBank, toBank, isl)

        writer.write(line)

formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)

# convert to federated learning format
data = pd.read_csv(outPath, index_col=False)

#generate bank clusters
bank_clusters = cluster_banks(data, "FromBank", 10)

data["bankId"] = data["FromBank"]

data["bankId_rank"] = data["bankId"].map(bank_clusters)

bank_to_cluster = {bank: cluster_num for cluster_num, banks in bank_clusters.items() for bank in banks}
data["bankId_rank"] = data["bankId"].map(bank_to_cluster)

data =data.drop(columns=["bankId","FromBank", "ToBank"])

mapping_ranks = {rank: i for i, rank in enumerate(data["bankId_rank"].unique())}

data["bankId_rank"] = data["bankId_rank"].map(mapping_ranks)

print("Unique partitions after clustering: ", data["bankId_rank"].unique())

fl_outPath = os.path.dirname(inPath) + "/formatted_transactions_fl.csv"
data.to_csv(fl_outPath, index=False)