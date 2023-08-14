use aide::axum::{
	routing::{delete, get, post, put},
	ApiRouter,
};
use axum::{extract::Path, http::StatusCode, Extension};
use axum_jsonschema::Json;
use schemars::JsonSchema;
use std::time::Instant;

use crate::{
	db::{self, Collection, DbExtension, Embedding, Error as DbError, SimilarityResult, MetadataEqualities},
	errors::HTTPError,
	similarity::Distance,
};

pub fn handler() -> ApiRouter {
	ApiRouter::new().nest(
		"/collections",
		ApiRouter::new()
			.api_route("/:collection_name", put(create_collection))
			.api_route("/:collection_name", post(query_collection))
			.api_route("/:collection_name", get(get_collection_info))
			.api_route("/:collection_name", delete(delete_collection))
			.api_route("/:collection_name/insert", post(insert_into_collection))
			.api_route("/:collection_name/:id", get(query_id_collection))
			.api_route("/:collection_name/:id", delete(delete_id_collection))
			.api_route("/:collection_name/query", post(query_metadata_string_collection))
			.api_route("/:collection_name/querynum", post(query_metadata_number_collection))
	)
}

/// Create a new collection
async fn create_collection(
	Path(collection_name): Path<String>,
	Extension(db): DbExtension,
	Json(req): Json<Collection>,
) -> Result<StatusCode, HTTPError> {
	tracing::trace!(
		"Creating collection {collection_name} with dimension {}",
		req.dimension
	);

	let mut db = db.write().await;

	let create_result = db.create_collection(collection_name, req.dimension, req.distance);

	match create_result {
		Ok(_) => Ok(StatusCode::CREATED),
		Err(db::Error::UniqueViolation) => {
			Err(HTTPError::new("Collection already exists").with_status(StatusCode::CONFLICT))
		},
		Err(_) => Err(HTTPError::new("Couldn't create collection")),
	}
}

#[derive(Debug, serde::Deserialize, JsonSchema)]
struct QueryCollectionQuery {
	/// Vector to query with
	query: Vec<f32>,
	/// Number of results to return
	k: Option<usize>,
}

/// Query a collection
#[allow(clippy::significant_drop_tightening)]
async fn query_collection(
	Path(collection_name): Path<String>,
	Extension(db): DbExtension,
	Json(req): Json<QueryCollectionQuery>,
) -> Result<Json<Vec<SimilarityResult>>, HTTPError> {
	tracing::trace!("Querying collection {collection_name}");

	let db = db.read().await;
	let collection = db
		.get_collection(&collection_name)
		.ok_or_else(|| HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))?;

	if req.query.len() != collection.dimension {
		return Err(HTTPError::new("Query dimension mismatch").with_status(StatusCode::BAD_REQUEST));
	}

	let instant = Instant::now();
	let results = collection.get_similarity(&req.query, req.k.unwrap_or(1));


	tracing::trace!("Query to {collection_name} took {:?}", instant.elapsed());
	Ok(Json(results))
}

#[derive(Debug, serde::Serialize, JsonSchema)]
struct CollectionInfo {
	/// Name of the collection
	name: String,
	/// Dimension of the embeddings in the collection
	dimension: usize,
	/// Distance function used for the collection
	distance: Distance,
	/// Number of embeddings in the collection
	embedding_count: usize,
}

/// Get collection info
#[allow(clippy::significant_drop_tightening)]
async fn get_collection_info(
	Path(collection_name): Path<String>,
	Extension(db): DbExtension,
) -> Result<Json<CollectionInfo>, HTTPError> {
	tracing::trace!("Getting collection info for {collection_name}");

	let db = db.read().await;
	let collection = db
		.get_collection(&collection_name)
		.ok_or_else(|| HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))?;

	Ok(Json(CollectionInfo {
		name: collection_name,
		distance: collection.distance,
		dimension: collection.dimension,
		embedding_count: collection.embeddings.len(),
	}))
}

/// Delete a collection
async fn delete_collection(
	Path(collection_name): Path<String>,
	Extension(db): DbExtension,
) -> Result<StatusCode, HTTPError> {
	tracing::trace!("Deleting collection {collection_name}");

	let mut db = db.write().await;

	let delete_result = db.delete_collection(&collection_name);

	match delete_result {
		Ok(_) => Ok(StatusCode::NO_CONTENT),
		Err(DbError::NotFound) => {
			Err(HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))
		},
		Err(_) => Err(HTTPError::new("Couldn't delete collection")),
	}
}

/// Insert a vector into a collection
async fn insert_into_collection(
	Path(collection_name): Path<String>,
	Extension(db): DbExtension,
	Json(embedding): Json<Embedding>,
) -> Result<StatusCode, HTTPError> {
	tracing::trace!("Inserting into collection {collection_name}");

	let mut db = db.write().await;

	let insert_result = db.insert_into_collection(&collection_name, embedding);

	match insert_result {
		Ok(_) => Ok(StatusCode::CREATED),
		Err(DbError::NotFound) => {
			Err(HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))
		},
		Err(DbError::UniqueViolation) => {
			Err(HTTPError::new("Vector already exists").with_status(StatusCode::CONFLICT))
		},
		Err(DbError::DimensionMismatch) => Err(HTTPError::new(
			"The provided vector has the wrong dimension",
		).with_status(StatusCode::BAD_REQUEST)),
		Err(_)=>Err(HTTPError::new(
			"Unknown Error",
		).with_status(StatusCode::BAD_REQUEST)),
	}
}

async fn query_id_collection(
	Path((collection_name, id)): Path<(String, String)>,
	Extension(db): DbExtension,
) -> Result<Json<Embedding>, HTTPError> {
	tracing::trace!("Getting query info for {id} in {collection_name}");

	let db = db.read().await;
	let collection = db
		.get_collection(&collection_name)
		.ok_or_else(|| HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))?;

	let instant = Instant::now();
	let result = collection.get_id(&id);

	tracing::trace!("Query ID {id} for {collection_name} took {:?}", instant.elapsed());
	match result {
		Some(embed) => Ok(Json(embed)),
		None => Err(HTTPError::new("No item found of ID").with_status(StatusCode::BAD_REQUEST))
	}
}

async fn delete_id_collection(
	Path((collection_name, id)): Path<(String, String)>,
	Extension(db): DbExtension,
) -> Result<StatusCode, HTTPError> {
	tracing::trace!("Deleting id {id} from {collection_name}");

	let mut db = db.write().await;

	//let delete_result = db.delete_collection(&collection_name);
	let delete_result: Result<Embedding, DbError> = db.collection_delete_id(&collection_name, &id);

	match delete_result {
		Ok(_) => Ok(StatusCode::NO_CONTENT),
		Err(DbError::NotFound) => {
			Err(HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))
		},
		Err(DbError::IDNotFound) => {
			Err(HTTPError::new("ID not found within specified collection").with_status(StatusCode::NOT_FOUND))
		},
		Err(_) => Err(HTTPError::new("Couldn't delete ID")),
	}
}

#[derive(Debug, serde::Deserialize, JsonSchema)]
struct QueryMetadataString{
	key: String,
	value: String,
	k: Option<usize>,
}

async fn query_metadata_string_collection(
	Path(collection_name): Path<String>,
	Extension(db): DbExtension,
	Json(req): Json<QueryMetadataString>,
) -> Result<Json<Vec<Embedding>>, HTTPError> {
	tracing::trace!("Metadata query for {collection_name}");

	let db = db.read().await;
	let collection = db
		.get_collection(&collection_name)
		.ok_or_else(|| HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))?;

	let instant = Instant::now();
	let result = collection.get_metadata_string(&req.key, &req.value, req.k.unwrap_or(5));

	tracing::trace!("Metadata Query for {collection_name} took {:?}", instant.elapsed());
	Ok(Json(result))
}


#[derive(Debug, serde::Deserialize, JsonSchema)]
struct QueryMetadataNumber{
	key: String,
	value: f32,
	equality: String,
	k: Option<usize>,
}

async fn query_metadata_number_collection(
	Path(collection_name): Path<String>,
	Extension(db): DbExtension,
	Json(req): Json<QueryMetadataNumber>,
) -> Result<Json<Vec<Embedding>>, HTTPError> {
	tracing::trace!("Metadata query for {collection_name}");

	let db = db.read().await;
	let collection = db
		.get_collection(&collection_name)
		.ok_or_else(|| HTTPError::new("Collection not found").with_status(StatusCode::NOT_FOUND))?;

	let instant = Instant::now();
	let eq = match MetadataEqualities::from_str(&req.equality.as_str()){
		Some(eq) => eq,
		None => return Err(HTTPError::new("Invalid equality string. Acceptable inputs; greater_than, greater_equal_than, lesser_than, lesser_equal_than, equal").with_status(StatusCode::BAD_REQUEST))
	};
	let result = collection.get_metadata_number(&req.key, req.value, eq, req.k.unwrap_or(5));

	tracing::trace!("Metadata Query for {collection_name} took {:?}", instant.elapsed());
	Ok(Json(result))
}
