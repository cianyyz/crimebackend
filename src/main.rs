#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use anyhow::Result;
use tracing_subscriber::{
	prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};
#[cfg(feature = "llm")]
use std::path::PathBuf;
#[cfg(feature = "llm")]
use clap::Parser;


mod db;
mod errors;
mod routes;
mod server;
mod shutdown;
mod similarity;
#[cfg(feature = "llm")]
mod rustllm;


#[cfg(feature = "llm")]
#[derive(Parser)]
pub struct LLMModelArgs {
    model_architecture: Option<llm::ModelArchitecture>,
    model_path: Option<PathBuf>,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
}

#[cfg(feature = "llm")]
impl LLMModelArgs {
    pub fn available(&self) -> bool {
        match(&self.model_architecture, &self.model_path){
            (Some(_), Some(_)) => true,
            (_, None) => false,
            (None, _) => false
        }
    }
    pub fn to_tokenizer_source(&self) -> llm::TokenizerSource {
        match (&self.tokenizer_path, &self.tokenizer_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --tokenizer-path and --tokenizer-repository");
            }
            (Some(path), None) => llm::TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::TokenizerSource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::TokenizerSource::Embedded,
        }
    }
}

#[cfg(feature = "llm")]
#[tokio::main]
async fn main() -> Result<()> {
	let args = LLMModelArgs::parse();
	tracing_subscriber::registry()
		.with(tracing_subscriber::fmt::layer().with_filter(
			EnvFilter::try_from_default_env().unwrap_or_else(|_| "tinyvector=info".into()),
		))
		.init();
	server::start(args).await
}

#[cfg(not(feature = "llm"))]
#[tokio::main]
async fn main() -> Result<()> {
	tracing_subscriber::registry()
		.with(tracing_subscriber::fmt::layer().with_filter(
			EnvFilter::try_from_default_env().unwrap_or_else(|_| "tinyvector=info".into()),
		))
		.init();
	server::start().await
}
