
use axum::Extension;
use rayon::prelude::*;
use schemars::JsonSchema;
use anyhow::Context;
use lazy_static::lazy_static;
use std::{
	collections::{BinaryHeap, HashMap},
	fs::{self},
	path::PathBuf,
	sync::Arc,
};
use tokio::sync::RwLock;

use crate::similarity::{get_cache_attr, get_distance_fn, normalize, Distance, ScoreIndex};

lazy_static! {
	pub static ref STORE_PATH: PathBuf = PathBuf::from("./storage/db");
}

#[allow(clippy::module_name_repetitions)]
pub type DbExtension = Extension<Arc<RwLock<Db>>>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Collection already exists")]
	UniqueViolation,

	#[error("Collection doesn't exist")]
	NotFound,

	#[error("The dimension of the vector doesn't match the dimension of the collection")]
	DimensionMismatch,

	#[error("ID doesn't exist within collection")]
	IDNotFound
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Db {
	pub collections: HashMap<String, Collection>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct SimilarityResult {
	score: f32,
	embedding: Embedding,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema)]
pub enum MetadataEqualities{
	GreaterEqualThan,
	GreaterThan,
	LesserEqualThan,
	LesserThan,
	Equal
}

impl MetadataEqualities {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "greater_equal_than" => Some(MetadataEqualities::GreaterEqualThan),
            "greater_than" => Some(MetadataEqualities::GreaterThan),
            "lesser_equal_than" => Some(MetadataEqualities::LesserEqualThan),
            "lesser_than" => Some(MetadataEqualities::LesserThan),
            "equal" => Some(MetadataEqualities::Equal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct Collection {
	/// Dimension of the vectors in the collection
	pub dimension: usize,
	/// Distance metric used for querying
	pub distance: Distance,
	/// Embeddings in the collection
	#[serde(default)]
	pub embeddings: Vec<Embedding>,
}

impl Collection {
	pub fn get_id(&self, id: &String) -> Option<Embedding>{
		self.embeddings
		.iter()
		.find(|embedding| &embedding.id == id)
		.cloned()
	}

	pub fn delete_id(&mut self, id: &String) -> Result<Embedding, Error>{
		 if let Some(index) = self.embeddings.iter().position(|embedding| &embedding.id == id) {
            // Remove the embedding from the vector and return it
            Ok(self.embeddings.remove(index))
        } else {
            // If the id is not found, return an error
            return Err(Error::IDNotFound);
        }
	}
	pub fn get_metadata_string(&self, key: &String, value: &String, k: usize) -> Vec<Embedding>{
		let filtered_embeddings: Vec<Embedding> = self.embeddings
            .iter()
            .filter(|embedding| {
                if let Some(metadata) = &embedding.metadata {
                    if let Some(meta_value) = metadata.get(key) {
                        return meta_value == value;
                    }
                }
                false
            })
            .cloned()
            .collect();
		
		filtered_embeddings.into_iter().take(k).collect()
    }

	pub fn get_metadata_number(&self, key: &str, value: f32, equality: MetadataEqualities, k: usize) -> Vec<Embedding> {
        // Filter embeddings based on the specified key and value comparison
        let filtered_embeddings: Vec<Embedding> =  self.embeddings
            .iter()
            .filter(|embedding| {
                if let Some(metadata) = &embedding.metadata {
                    if let Some(meta_value_str) = metadata.get(key) {
                        if let Ok(meta_value) = meta_value_str.parse::<f32>() {
							match equality {
								MetadataEqualities::GreaterEqualThan => {return meta_value >= value;},
								MetadataEqualities::GreaterThan => {return meta_value > value;},
								MetadataEqualities::LesserEqualThan => {return meta_value <= value;}
								MetadataEqualities::LesserThan => {return meta_value < value;}
								MetadataEqualities::Equal => {return meta_value == value;}
							}
                            
                        }
                    }
                }
                false
            })
            .cloned()
            .collect();
		
		filtered_embeddings.into_iter().take(k).collect()
    }

	pub fn get_similarity(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
		let memo_attr = get_cache_attr(self.distance, query);
		let distance_fn = get_distance_fn(self.distance);

		let scores = self
			.embeddings
			.par_iter()
			.enumerate()
			.map(|(index, embedding)| {
				let score = distance_fn(&embedding.vector, query, memo_attr);
				ScoreIndex { score, index }
			})
			.collect::<Vec<_>>();

		let mut heap = BinaryHeap::new();
		for score_index in scores {
			if heap.len() < k || score_index < *heap.peek().unwrap() {
				heap.push(score_index);

				if heap.len() > k {
					heap.pop();
				}
			}
		}

		heap.into_sorted_vec()
			.into_iter()
			.map(|ScoreIndex { score, index }| SimilarityResult {
				score,
				embedding: self.embeddings[index].clone(),
			})
			.collect()
	}
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct Embedding {
	pub id: String,
	pub vector: Vec<f32>,
	pub metadata: Option<HashMap<String, String>>,
}

impl Db {
	pub fn new() -> Self {
		Self {
			collections: HashMap::new(),
		}
	}

	pub fn extension(self) -> DbExtension {
		Extension(Arc::new(RwLock::new(self)))
	}

	pub fn create_collection(
		&mut self,
		name: String,
		dimension: usize,
		distance: Distance,
	) -> Result<Collection, Error> {
		if self.collections.contains_key(&name) {
			return Err(Error::UniqueViolation);
		}

		let collection = Collection {
			dimension,
			distance,
			embeddings: Vec::new(),
		};

		self.collections.insert(name, collection.clone());
		self.save();
		Ok(collection)
	}

	pub fn delete_collection(&mut self, name: &str) -> Result<(), Error> {
		if !self.collections.contains_key(name) {
			return Err(Error::NotFound);
		}

		self.collections.remove(name);
		self.save();
		Ok(())
	}

	pub fn insert_into_collection(
		&mut self,
		collection_name: &str,
		mut embedding: Embedding,
	) -> Result<(), Error> {
		let collection = self
			.collections
			.get_mut(collection_name)
			.ok_or(Error::NotFound)?;

		if embedding.vector.len() != collection.dimension {
			return Err(Error::DimensionMismatch);
		}

		// Normalize the vector if the distance metric is cosine, so we can use dot product later
		if collection.distance == Distance::Cosine {
			embedding.vector = normalize(&embedding.vector);
		}

		if collection.embeddings.iter().any(|e| e.id == embedding.id) {
			let _ = collection.delete_id(&embedding.id);
		}

		collection.embeddings.push(embedding);
		self.save();
		Ok(())
	}

	pub fn collection_delete_id(&mut self, collection_name: &str, id: &String) -> Result<Embedding, Error>{
		let collection = self
			.collections
			.get_mut(collection_name)
			.ok_or(Error::NotFound)?;
		let result = collection.delete_id(id);
		self.save();
		result
	}


	pub fn get_collection(&self, name: &str) -> Option<&Collection> {
		self.collections.get(name)
	}

	fn load_from_store() -> anyhow::Result<Self> {
		if !STORE_PATH.exists() {
			tracing::debug!("Creating database store");
			fs::create_dir_all(STORE_PATH.parent().context("Invalid store path")?)?;

			return Ok(Self::new());
		}

		tracing::debug!("Loading database from store");
		let db = fs::read(STORE_PATH.as_path())?;
		Ok(bincode::deserialize(&db[..])?)
	}

	fn save_to_store(&self) -> anyhow::Result<()> {
		let db = bincode::serialize(self)?;

		fs::write(STORE_PATH.as_path(), db)?;

		Ok(())
	}

	pub fn save(&self){
		self.save_to_store().ok();
	}
}

impl Drop for Db {
	fn drop(&mut self) {
		tracing::info!("Saving database to store");
		self.save_to_store().ok();
	}
}

pub fn from_store() -> anyhow::Result<Db> {
	Db::load_from_store()
}
