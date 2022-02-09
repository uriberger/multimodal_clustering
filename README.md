# Multimodal Self-Supervised Clustering

Our model consists of two encoders (a visual encoder and a text encoder), both of which encode relevant inputs into a binary discrete space interpreted as multi-label clustering of the inputs. The model is initialized randomly, without pre-training, and is trained in a self-supervised manner using (image, caption) pairs.



## Visualization of the Model
![Alt text](model_desc.png)
