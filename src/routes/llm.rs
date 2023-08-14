use aide::axum::{routing::post, ApiRouter};
use axum::{http::StatusCode, Extension};
use axum_jsonschema::Json;
use schemars::JsonSchema;

use crate::{
    rustllm::LLMExtension,
	errors::HTTPError,
};

pub fn handler() -> ApiRouter {
	ApiRouter::new()
		.route("/llm", post(query_prompt))
}

#[derive(Debug, serde::Deserialize, JsonSchema)]
struct PromptQuery {
    pub query: String,
}

/// Query a collection
#[allow(clippy::significant_drop_tightening)]
async fn query_prompt(
	Extension(model): LLMExtension,
	Json(req): Json<PromptQuery>,
) -> Result<Json<String>, HTTPError> {
    let query = req.query;
    let now = std::time::Instant::now();
	tracing::trace!("Getting embeddings for {query}");
    let model = model.write().await;
    let inf_result = model.inference(query.as_str());
    tracing::info!("\nInference Time: {}ms", now.elapsed().as_millis());
    match inf_result {
        Ok(result) => Ok(Json(result)),
        Err(_) => return Err(HTTPError::new("Inference Error").with_status(StatusCode::BAD_REQUEST))
    }
}

