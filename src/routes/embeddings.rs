use aide::{
	axum::{routing::{post}, ApiRouter},
};
use axum::{Extension};
use axum_jsonschema::Json;
use schemars::JsonSchema;

use crate::{
    rustllm::LLMExtension,
	errors::HTTPError,
};

pub fn handler() -> ApiRouter {
	ApiRouter::new()
		.route("/embeddings", post(query_embeddings))
}

#[derive(Debug, serde::Deserialize, JsonSchema)]
struct EmbeddingsQuery {
    pub query: String,
}

/// Query a collection
#[allow(clippy::significant_drop_tightening)]
async fn query_embeddings(
	Extension(emb): LLMExtension,
	Json(req): Json<EmbeddingsQuery>,
) -> Result<Json<Vec<f32>>, HTTPError> {
    let query = req.query;
	tracing::trace!("Getting embeddings for {query}");
    let emb = emb.write().await;
    let embeddings: Vec<f32> = emb.get_embeddings(query.as_str());
	Ok(Json(embeddings))
}

