from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    url="https://a873a436-c77e-47ad-8632-cc49d03295a8.eu-central-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.OaaGGkLXWpsWRopUvwmOWZzpD456F-Vvl1hICzmuvaA"
)

client.recreate_collection(
    collection_name="memory_bot",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)