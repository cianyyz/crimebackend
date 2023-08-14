use aide::axum::ApiRouter;

mod collection;
mod docs;
mod system;
#[cfg(feature = "llm")]
mod embeddings;
#[cfg(feature = "llm")]
mod llm;

#[cfg(feature = "llm")]
pub fn handler() -> ApiRouter {
	ApiRouter::new()
		.merge(docs::handler())
		.merge(system::handler())
		.merge(collection::handler())
		.merge(embeddings::handler())
		.merge(llm::handler())
}

#[cfg(not(feature = "llm"))]
pub fn handler() -> ApiRouter {
	ApiRouter::new()
		.merge(docs::handler())
		.merge(system::handler())
		.merge(collection::handler())
}