use aide::axum::ApiRouter;

mod collection;
mod docs;
mod system;
mod embeddings;
mod llm;

pub fn handler() -> ApiRouter {
	ApiRouter::new()
		.merge(docs::handler())
		.merge(system::handler())
		.merge(collection::handler())
		.merge(embeddings::handler())
		.merge(llm::handler())
}
