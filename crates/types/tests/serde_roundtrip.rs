use kbolt_types::{
    Locator, OmitReason, SearchMode, SearchRequest, SearchResponse, SearchResult, SearchSignals,
};

#[test]
fn search_request_roundtrip() {
    let req = SearchRequest {
        query: "rust trait object".to_string(),
        mode: SearchMode::Auto,
        space: Some("work".to_string()),
        collections: vec!["api".to_string(), "docs".to_string()],
        limit: 10,
        min_score: 0.0,
        no_rerank: false,
        debug: true,
    };

    let json = serde_json::to_string(&req).expect("serialize SearchRequest");
    let decoded: SearchRequest = serde_json::from_str(&json).expect("deserialize SearchRequest");
    assert_eq!(decoded, req);
}

#[test]
fn locator_and_omit_reason_roundtrip() {
    let locator = Locator::DocId("a1b2c3".to_string());
    let locator_json = serde_json::to_string(&locator).expect("serialize Locator");
    let decoded_locator: Locator =
        serde_json::from_str(&locator_json).expect("deserialize Locator");
    assert_eq!(decoded_locator, locator);

    let reason = OmitReason::MaxBytes;
    let reason_json = serde_json::to_string(&reason).expect("serialize OmitReason");
    let decoded_reason: OmitReason =
        serde_json::from_str(&reason_json).expect("deserialize OmitReason");
    assert_eq!(decoded_reason, reason);
}

#[test]
fn search_response_roundtrip() {
    let response = SearchResponse {
        results: vec![SearchResult {
            docid: "#a1b2c3".to_string(),
            path: "api/src/lib.rs".to_string(),
            title: "Engine".to_string(),
            space: "work".to_string(),
            collection: "api".to_string(),
            heading: Some("# Engine".to_string()),
            text: "pub struct Engine {}".to_string(),
            score: 0.88,
            signals: Some(SearchSignals {
                bm25: Some(0.52),
                dense: Some(0.79),
                rrf: 0.62,
                reranker: Some(0.91),
            }),
        }],
        query: "engine struct".to_string(),
        mode: SearchMode::Auto,
        staleness_hint: Some("Index last updated: 2m ago".to_string()),
        elapsed_ms: 24,
    };

    let json = serde_json::to_string(&response).expect("serialize SearchResponse");
    let decoded: SearchResponse = serde_json::from_str(&json).expect("deserialize SearchResponse");
    assert_eq!(decoded, response);
}
