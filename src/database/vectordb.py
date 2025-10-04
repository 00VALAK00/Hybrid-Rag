from qdrant_client import QdrantClient
from qdrant_client.models import SparseVectorParams,VectorParams,SparseVector, Prefetch, Distance, PointStruct, MultiVectorConfig, MultiVectorComparator,HnswConfigDiff,Modifier
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding  
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
print(root_dir)
if str(root_dir) not in sys.path:       
     sys.path.insert(0,str(root_dir))

from utils.data_chunking import load_and_chunk_pdfs
import logging
import time

logging.basicConfig(level=logging.INFO)

api_key  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Pvk07_ZxPmNQfhCrubw_QUxMr0RWcQexJCFtjQ0iDAc"
url = "https://546d316b-4363-4247-b27c-e98adbf3aec6.us-west-1-0.aws.cloud.qdrant.io"

class QdrantVectorDB:

    def __init__(self, collection="docs"):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=360)
        self.collection = collection

        self.dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        self.late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

        if not self.client.collection_exists(self.collection):
            logging.info(f"Collection {self.collection} was not found. Creating Now...")

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense_embeddings": 
                        VectorParams(size=self.dense_embedding_model.embedding_size, 
                                    distance=Distance.COSINE
                                    ),
                    "late_interaction_embeddings": 
                                    VectorParams(size=self.late_interaction_embedding_model.embedding_size,
                                    distance=Distance.COSINE,
                                    multivector_config= MultiVectorConfig(
                                        comparator=MultiVectorComparator.MAX_SIM
                                        ),
                                    hnsw_config=HnswConfigDiff(m=0) # disable indexing for the reranker
                                    ),
                },   
                sparse_vectors_config={
                    "bm25": SparseVectorParams(modifier=Modifier.IDF)
                }
                
            )
        logging.info(f"Collection {self.collection} was found. Skipping creation")


    def embed_chunks(self, chunks: list[dict])-> list[PointStruct]:
            logging.info("Started embedding the documents")
            start = time.time()
            bm25_embeddings = list(self.bm25_embedding_model.embed(chunk["content"] for chunk in chunks))
            late_interaction_embeddings = list(self.late_interaction_embedding_model.embed(chunk["content"] for chunk in chunks))
            dense_embeddings = list(self.dense_embedding_model.embed(chunk["content"] for chunk in chunks))

            points = []
            for i, (payload,bm25_embedding,late_interaction_embedding,dense_embedding) in enumerate(zip(chunks, bm25_embeddings,late_interaction_embeddings,dense_embeddings)):
                 
                 points.append(PointStruct(
                    id=i,
                    vector = {
                            "dense_embeddings": dense_embedding,
                            "bm25":bm25_embedding.as_object(),
                            "late_interaction_embeddings":late_interaction_embedding,
                    },
                    payload = payload,
                ))
            end = time.time()
            logging.info(f"Successfully chunked the documents. Process took {(end-start)/60:.2f} s")
            return points
        
                
    def upsertData(self):
        chunks = load_and_chunk_pdfs()
        points = self.embed_chunks(chunks=chunks)

        logging.info("Started Upserting data to the db")
        
        self.client.upload_points(
            collection_name=self.collection,
            points= points,
            batch_size=32,
            wait=True

        )
        logging.info("Successfully ingested the data")
        return points

            
            
    def search(self, query:str, topk = 5) -> list[str]:
         # embedding the query
        bm25_query_embedding = next(self.bm25_embedding_model.query_embed(query=query))
        dense_query_embedding = next(self.dense_embedding_model.query_embed(query=query))
        lie_query_embedding = next(self.late_interaction_embedding_model.query_embed(query=query))

        prefetch = [
            Prefetch(
                query=dense_query_embedding,
                using="dense_embeddings",
                limit=20,
            ),
            Prefetch(
                query=SparseVector(**bm25_query_embedding.as_object()),
                using="bm25",
                limit=20,
        )]

        results = self.client.query_points(
        self.collection,
        prefetch=prefetch,
        query=lie_query_embedding,
        using="late_interaction_embeddings",
        with_payload=True,
        limit=topk,
        ).points
        print(results)
        
        return [f"context{i+1}: "+ str(result.payload) + "\n\n" for i, result in enumerate(results)]
    