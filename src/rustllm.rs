
use axum::Extension;
use llm::{ModelArchitecture, Model};
use std::{
	path::PathBuf,
	sync::Arc,
};
use tokio::sync::RwLock;
use std::{convert::Infallible, io::Write};

use crate::LLMModelArgs;


#[allow(clippy::module_name_repetitions)]
pub type LLMExtension = Extension<Arc<RwLock<LLMModel>>>;

pub struct LLMModel {
    pub inference_parameters: llm::InferenceParameters,
    pub model: Box<dyn Model> 
}

impl LLMModel {
	pub fn new(args: LLMModelArgs) -> Self {
        let tokenizer_source: llm::TokenizerSource = args.to_tokenizer_source();
        let model_architecture: ModelArchitecture = args.model_architecture.unwrap();
        let model_path: PathBuf = args.model_path.unwrap();
        let model_params: llm::ModelParameters = llm::ModelParameters::default();
        let inference_parameters: llm::InferenceParameters = llm::InferenceParameters::default();
        let model: Box<dyn Model> = llm::load_dynamic(
            Some(model_architecture),
            &model_path,
            tokenizer_source,
            model_params,
            llm::load_progress_callback_stdout,
        ).unwrap_or_else(|err| {
            panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
        });
		Self {
            inference_parameters,
            model
		}
	}

	pub fn extension(self) -> LLMExtension {
		Extension(Arc::new(RwLock::new(self)))
	}

    pub fn get_embeddings(
        &self,
        query: &str,
    ) -> Vec<f32> {
        let mut session = self.model.start_session(Default::default());
        let mut output_request = llm::OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };
        let vocab = self.model.tokenizer();
        let beginning_of_sentence = true;
        let query_token_ids = vocab
            .tokenize(query, beginning_of_sentence)
            .unwrap()
            .iter()
            .map(|(_, tok)| *tok)
            .collect::<Vec<_>>();
        self.model.evaluate(&mut session, &query_token_ids, &mut output_request);
        output_request.embeddings.unwrap()
    }


    pub fn inference(&self, prompt: &str) -> Result<String,  llm::InferenceError> {
        let mut session = self.model.start_session(Default::default());
        let mut result = String::from("");
        let _  = session.infer::<Infallible>(
            self.model.as_ref(),
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: prompt.into(),
                parameters: &llm::InferenceParameters::default(),
                play_back_previous_tokens: false,
                maximum_token_count: None,
            },
            // OutputRequest
            &mut Default::default(),
            |r| match r {
                llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                    print!("{t}");
                    result.push_str(&t);
                    std::io::stdout().flush().unwrap();

                    Ok(llm::InferenceFeedback::Continue)
                }
                _ => Ok(llm::InferenceFeedback::Continue),
            },
        );
        println!("");
        Ok(result)

    }

	
}

impl Drop for LLMModel {
	fn drop(&mut self) {
		tracing::info!("Closing llm");
		//self.save_to_store().ok();
	}
}
