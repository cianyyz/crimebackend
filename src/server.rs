use aide::openapi::{self, OpenApi};
use anyhow::Result;
use axum::{Extension, Server};
use std::{env, net::SocketAddr};

use crate::{db, routes, shutdown::Shutdown, LLMModelArgs, rustllm::LLMModel};

#[allow(clippy::redundant_pub_crate)]
pub(crate) async fn start(args: LLMModelArgs) -> Result<()> {
	let mut openapi = OpenApi {
		info: openapi::Info {
			title: "Tinyvector".to_string(),
			version: env!("CARGO_PKG_VERSION").to_string(),
			..openapi::Info::default()
		},
		..OpenApi::default()
	};

	let db = db::from_store()?;
	let rustllm = LLMModel::new(args);
	let shutdown = Shutdown::new()?;
	let router = routes::handler().finish_api(&mut openapi);

	let router = router
		.layer(Extension(openapi))
		.layer(shutdown.extension())
		.layer(db.extension())
		.layer(rustllm.extension());

	let addr = SocketAddr::from((
		[0, 0, 0, 0],
		env::var("PORT").map_or(Ok(8000), |p| p.parse())?,
	));
	tracing::info!("Starting server on {addr}...");
	Server::bind(&addr)
		.serve(router.into_make_service())
		.with_graceful_shutdown(shutdown.handle())
		.await?;

	Ok(())
}
