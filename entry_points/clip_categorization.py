import clip
import torch
from dataset_builders.category_dataset import generate_fountain_category_dataset
from sklearn.cluster import KMeans, SpectralClustering
import networkx as nx
from chinese_whispers import chinese_whispers, aggregate_clusters
from metrics import CategorizationMetric
from dataset_builders.dataset_builder_creator import create_dataset_builder
from datasets_src.dataset_config import DatasetConfig


def clip_text_inference(model, inputs):
    model_inputs = clip.tokenize(inputs)
    return model.encode_text(model_inputs).float()


def clip_similarity_func(image_features, text_features):
    norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return norm_image_features @ norm_text_features.T


dataset_builder, _, _ = create_dataset_builder('COCO')
training_set_config = DatasetConfig(1)
training_set, _, _ = dataset_builder.build_dataset(training_set_config)
token_count = training_set.get_token_count()

category_dataset = generate_fountain_category_dataset()
word_lists = list(category_dataset.values())
all_words = [word for outer in word_lists for word in outer]
word_num = len(all_words)
print('All words num: ' + str(word_num))

all_words = [word for word in all_words if word in token_count]
word_num = len(all_words)
print('Words in MSCOCO num: ' + str(word_num))

model, _ = clip.load('RN50', torch.device('cpu'))
all_prompts = ['a photo of a ' + word for word in all_words]

emb_mat = clip_text_inference(model, all_prompts)
print(emb_mat.shape)

# sim_mat = clip_similarity_func(emb_mat, emb_mat)
# G = nx.Graph()
# G.add_nodes_from(range(word_num))
# edge_lists = [[(x, y, {'weight': sim_mat[x, y].item()}) for y in range(x+1, word_num)] for x in range(word_num)]
# all_edges = [edge for outer in edge_lists for edge in outer]
# for i in range(word_num - 1):
#     all_edges[i][2]['weight'] = 0
# G.add_edges_from(all_edges)
# chinese_whispers(G, weighting='top')
# print('ID\tCluster\n')
#
# for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
#     print('{}\t{}\n'.format(label, cluster))

kmeans = KMeans().fit(emb_mat.detach().numpy())
# kmeans = KMeans(n_clusters=41).fit(emb_mat.detach().numpy())
cluster_list = list(kmeans.labels_)
# sc = SpectralClustering(assign_labels='discretize').fit(emb_mat.detach().numpy())
# cluster_list = list(sc.labels_)
print(len(cluster_list))

predicted_labels = {all_words[i]: cluster_list[i] for i in range(len(all_words))}

metric = CategorizationMetric(None, category_dataset, predicted_labels)
print(metric.report())
