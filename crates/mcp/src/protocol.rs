use crate::{McpAdapter, McpToolDefinition};
use serde::Deserialize;
use serde_json::{json, Map, Value};

const JSONRPC_VERSION: &str = "2.0";
pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

const CODE_INVALID_REQUEST: i64 = -32600;
const CODE_METHOD_NOT_FOUND: i64 = -32601;
const CODE_INVALID_PARAMS: i64 = -32602;
const CODE_INTERNAL_ERROR: i64 = -32603;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    AwaitInitialize,
    AwaitInitialized,
    Ready,
}

pub struct McpProtocol {
    state: SessionState,
}

impl Default for McpProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl McpProtocol {
    pub fn new() -> Self {
        Self {
            state: SessionState::AwaitInitialize,
        }
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn handle_message(&mut self, adapter: &McpAdapter, message: Value) -> Option<Value> {
        let request = match parse_incoming_request(message) {
            Ok(request) => request,
            Err(error_response) => return Some(error_response),
        };

        let is_notification = request.id.is_none();

        match request.method.as_str() {
            "ping" => {
                if is_notification {
                    return None;
                }
                let id = request
                    .id
                    .expect("request id should exist for non-notification");
                Some(success_response(id, json!({})))
            }
            "initialize" => {
                if is_notification {
                    return None;
                }
                let id = request
                    .id
                    .expect("request id should exist for non-notification");
                if self.state != SessionState::AwaitInitialize {
                    return Some(error_response(
                        Some(id),
                        CODE_INVALID_REQUEST,
                        "initialize can only be called once at session start",
                        None,
                    ));
                }

                let params: InitializeParams = match parse_params("initialize", request.params) {
                    Ok(params) => params,
                    Err(err) => {
                        return Some(error_response(Some(id), CODE_INVALID_PARAMS, &err, None))
                    }
                };
                let _ = (&params.capabilities, &params.client_info);

                if params.protocol_version != MCP_PROTOCOL_VERSION {
                    return Some(error_response(
                        Some(id),
                        CODE_INVALID_PARAMS,
                        "Unsupported protocol version",
                        Some(json!({
                            "supported": [MCP_PROTOCOL_VERSION],
                            "requested": params.protocol_version,
                        })),
                    ));
                }

                let instructions = match adapter.dynamic_instructions() {
                    Ok(instructions) => instructions,
                    Err(err) => {
                        return Some(error_response(
                            Some(id),
                            CODE_INTERNAL_ERROR,
                            &format!("failed to build dynamic instructions: {err}"),
                            None,
                        ))
                    }
                };

                let result = json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    },
                    "serverInfo": {
                        "name": "kbolt",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "instructions": instructions
                });

                self.state = SessionState::AwaitInitialized;
                Some(success_response(id, result))
            }
            "notifications/initialized" => {
                if is_notification && self.state == SessionState::AwaitInitialized {
                    self.state = SessionState::Ready;
                }
                None
            }
            "tools/list" => {
                if !self.ready_for_operations() {
                    return if is_notification {
                        None
                    } else {
                        Some(error_response(
                            request.id,
                            CODE_INVALID_REQUEST,
                            "session is not ready; send notifications/initialized first",
                            None,
                        ))
                    };
                }
                if is_notification {
                    return None;
                }
                let id = request
                    .id
                    .expect("request id should exist for non-notification");

                let params: ToolsListParams = match parse_params("tools/list", request.params) {
                    Ok(params) => params,
                    Err(err) => {
                        return Some(error_response(Some(id), CODE_INVALID_PARAMS, &err, None))
                    }
                };
                let _cursor = params.cursor;

                let tools = McpAdapter::tool_definitions()
                    .into_iter()
                    .map(tool_definition_to_json)
                    .collect::<Vec<_>>();
                Some(success_response(id, json!({ "tools": tools })))
            }
            "tools/call" => {
                if !self.ready_for_operations() {
                    return if is_notification {
                        None
                    } else {
                        Some(error_response(
                            request.id,
                            CODE_INVALID_REQUEST,
                            "session is not ready; send notifications/initialized first",
                            None,
                        ))
                    };
                }
                if is_notification {
                    return None;
                }
                let id = request
                    .id
                    .expect("request id should exist for non-notification");

                let params: ToolsCallParams = match parse_params("tools/call", request.params) {
                    Ok(params) => params,
                    Err(err) => {
                        return Some(error_response(Some(id), CODE_INVALID_PARAMS, &err, None))
                    }
                };

                let arguments = params
                    .arguments
                    .unwrap_or_else(|| Value::Object(Map::new()));
                let result = match adapter.call_tool_json(&params.name, arguments) {
                    Ok(structured_content) => {
                        let serialized = match serde_json::to_string(&structured_content) {
                            Ok(value) => value,
                            Err(err) => {
                                return Some(error_response(
                                    Some(id),
                                    CODE_INTERNAL_ERROR,
                                    &format!("failed to encode tool result: {err}"),
                                    None,
                                ))
                            }
                        };
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": serialized
                                }
                            ],
                            "structuredContent": structured_content,
                            "isError": false
                        })
                    }
                    Err(err) => json!({
                        "content": [
                            {
                                "type": "text",
                                "text": err.to_string()
                            }
                        ],
                        "isError": true
                    }),
                };

                Some(success_response(id, result))
            }
            _ => {
                if is_notification {
                    None
                } else {
                    Some(error_response(
                        request.id,
                        CODE_METHOD_NOT_FOUND,
                        "Method not found",
                        None,
                    ))
                }
            }
        }
    }

    fn ready_for_operations(&self) -> bool {
        self.state == SessionState::Ready
    }
}

#[derive(Debug)]
struct IncomingRequest {
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeParams {
    protocol_version: String,
    capabilities: Value,
    client_info: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolsListParams {
    #[serde(default)]
    cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolsCallParams {
    name: String,
    #[serde(default)]
    arguments: Option<Value>,
}

fn parse_incoming_request(message: Value) -> std::result::Result<IncomingRequest, Value> {
    let Value::Object(mut object) = message else {
        return Err(error_response(
            None,
            CODE_INVALID_REQUEST,
            "Invalid Request: expected a JSON object",
            None,
        ));
    };

    let id = object.remove("id");
    if let Some(id_value) = id.as_ref() {
        if !is_valid_request_id(id_value) {
            return Err(error_response(
                None,
                CODE_INVALID_REQUEST,
                "Invalid Request: id must be string, number, or null",
                None,
            ));
        }
    }

    let Some(jsonrpc) = object
        .remove("jsonrpc")
        .and_then(|value| value.as_str().map(ToString::to_string))
    else {
        return Err(error_response(
            id,
            CODE_INVALID_REQUEST,
            "Invalid Request: missing jsonrpc version",
            None,
        ));
    };

    if jsonrpc != JSONRPC_VERSION {
        return Err(error_response(
            id,
            CODE_INVALID_REQUEST,
            "Invalid Request: jsonrpc must be '2.0'",
            None,
        ));
    }

    let Some(method) = object
        .remove("method")
        .and_then(|value| value.as_str().map(ToString::to_string))
    else {
        return Err(error_response(
            id,
            CODE_INVALID_REQUEST,
            "Invalid Request: missing method",
            None,
        ));
    };

    let params = object.remove("params");
    Ok(IncomingRequest { id, method, params })
}

fn parse_params<T>(method: &str, params: Option<Value>) -> std::result::Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    let normalized = params.unwrap_or_else(|| Value::Object(Map::new()));
    serde_json::from_value(normalized).map_err(|err| format!("Invalid params for {method}: {err}"))
}

fn is_valid_request_id(id: &Value) -> bool {
    id.is_null() || id.is_string() || id.is_number()
}

fn tool_definition_to_json(tool: McpToolDefinition) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    })
}

fn success_response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "result": result
    })
}

fn error_response(id: Option<Value>, code: i64, message: &str, data: Option<Value>) -> Value {
    let mut error = json!({
        "code": code,
        "message": message
    });
    if let Some(data) = data {
        error["data"] = data;
    }

    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id.unwrap_or(Value::Null),
        "error": error
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use kbolt_core::engine::Engine;
    use kbolt_types::AddCollectionRequest;
    use serde_json::{json, Value};
    use tempfile::tempdir;

    use super::{McpProtocol, SessionState, MCP_PROTOCOL_VERSION};
    use crate::test_support::with_isolated_xdg_dirs;
    use crate::McpAdapter;

    fn initialize_request() -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "0.1.0"
                }
            }
        })
    }

    fn initialize_then_ready(protocol: &mut McpProtocol, adapter: &McpAdapter) {
        let initialize = protocol
            .handle_message(adapter, initialize_request())
            .expect("initialize should return a response");
        assert_eq!(
            initialize["result"]["protocolVersion"],
            MCP_PROTOCOL_VERSION
        );

        let notification = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let response = protocol.handle_message(adapter, notification);
        assert!(
            response.is_none(),
            "initialized notification has no response"
        );
        assert_eq!(protocol.state(), SessionState::Ready);
    }

    #[test]
    fn initialize_moves_session_to_await_initialized() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();

            let response = protocol
                .handle_message(&adapter, initialize_request())
                .expect("initialize should return a response");

            assert_eq!(response["jsonrpc"], "2.0");
            assert_eq!(response["id"], 1);
            assert_eq!(response["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
            assert_eq!(response["result"]["serverInfo"]["name"], "kbolt");
            assert!(response["result"]["instructions"].is_string());
            assert_eq!(protocol.state(), SessionState::AwaitInitialized);
        });
    }

    #[test]
    fn tools_operations_require_initialized_notification() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();

            protocol
                .handle_message(&adapter, initialize_request())
                .expect("initialize should return a response");
            assert_eq!(protocol.state(), SessionState::AwaitInitialized);

            let early_list = json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            });
            let early_response = protocol
                .handle_message(&adapter, early_list)
                .expect("tools/list should return a response");
            assert_eq!(early_response["error"]["code"], -32600);

            let initialized = json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            });
            protocol.handle_message(&adapter, initialized);
            assert_eq!(protocol.state(), SessionState::Ready);

            let list = json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list",
                "params": {}
            });
            let list_response = protocol
                .handle_message(&adapter, list)
                .expect("tools/list should return a response");
            let tools = list_response["result"]["tools"]
                .as_array()
                .expect("tools should be an array");
            assert_eq!(tools.len(), 5);
        });
    }

    #[test]
    fn tools_call_reports_tool_errors_with_is_error_true() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();
            initialize_then_ready(&mut protocol, &adapter);

            let call = json!({
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": ""
                    }
                }
            });

            let response = protocol
                .handle_message(&adapter, call)
                .expect("tools/call should return a response");
            assert_eq!(response["result"]["isError"], true);
            assert_eq!(response["result"]["content"][0]["type"], "text");
        });
    }

    #[test]
    fn unsupported_protocol_version_returns_invalid_params() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();

            let request = json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "0.1.0"
                    }
                }
            });

            let response = protocol
                .handle_message(&adapter, request)
                .expect("initialize should return a response");
            assert_eq!(response["error"]["code"], -32602);
            assert_eq!(
                response["error"]["data"]["supported"][0],
                MCP_PROTOCOL_VERSION
            );
            assert_eq!(protocol.state(), SessionState::AwaitInitialize);
        });
    }

    #[test]
    fn ping_is_allowed_before_initialization() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();

            let ping = json!({
                "jsonrpc": "2.0",
                "id": "p1",
                "method": "ping"
            });
            let response = protocol
                .handle_message(&adapter, ping)
                .expect("ping should return a response");
            assert_eq!(response["result"], json!({}));
        });
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();
            initialize_then_ready(&mut protocol, &adapter);

            let request = json!({
                "jsonrpc": "2.0",
                "id": 7,
                "method": "unknown/method",
                "params": {}
            });
            let response = protocol
                .handle_message(&adapter, request)
                .expect("unknown method should return a response");
            assert_eq!(response["error"]["code"], -32601);
        });
    }

    #[test]
    fn invalid_jsonrpc_version_is_rejected() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();

            let request = json!({
                "jsonrpc": "1.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            });
            let response = protocol
                .handle_message(&adapter, request)
                .expect("invalid request should return an error response");
            assert_eq!(response["error"]["code"], -32600);
        });
    }

    #[test]
    fn tools_call_can_return_structured_content() {
        with_isolated_xdg_dirs(|| {
            let root = tempdir().expect("create collection root");
            let engine = Engine::new(None).expect("create engine");
            engine.add_space("work", None).expect("add work");

            let work_path = root.path().join("work-api");
            fs::create_dir_all(&work_path).expect("create collection directory");
            engine
                .add_collection(AddCollectionRequest {
                    path: work_path.clone(),
                    space: Some("work".to_string()),
                    name: Some("api".to_string()),
                    description: None,
                    extensions: None,
                    no_index: true,
                })
                .expect("add work collection");
            fs::write(work_path.join("a.md"), "search-token\n").expect("write file");

            let adapter = McpAdapter::new(engine);
            adapter
                .update(kbolt_types::UpdateOptions {
                    space: Some("work".to_string()),
                    collections: vec!["api".to_string()],
                    no_embed: true,
                    dry_run: false,
                    verbose: false,
                })
                .expect("run update");

            let mut protocol = McpProtocol::new();
            initialize_then_ready(&mut protocol, &adapter);

            let call = json!({
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": "search-token",
                        "space": "work",
                        "collection": "api"
                    }
                }
            });

            let response = protocol
                .handle_message(&adapter, call)
                .expect("tools/call should return a response");
            assert_eq!(response["result"]["isError"], false);
            assert!(response["result"]["structuredContent"]["results"].is_array());
            assert_eq!(
                response["result"]["structuredContent"]["results"][0]["space"],
                "work"
            );
        });
    }

    #[test]
    fn parses_valid_request_id_types_only() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let mut protocol = McpProtocol::new();

            let invalid_id_request = json!({
                "jsonrpc": "2.0",
                "id": { "nested": true },
                "method": "ping"
            });
            let response = protocol
                .handle_message(&adapter, invalid_id_request)
                .expect("invalid id should return an error response");
            assert_eq!(response["error"]["code"], -32600);
        });
    }
}
