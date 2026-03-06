use std::io::{BufRead, BufReader, BufWriter, Write};

use kbolt_core::Result;
use serde_json::{json, Value};

use crate::protocol::McpProtocol;
use crate::McpAdapter;

const CODE_PARSE_ERROR: i64 = -32700;

pub fn run_stdio(adapter: &McpAdapter) -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let reader = BufReader::new(stdin.lock());
    let writer = BufWriter::new(stdout.lock());
    run_stdio_with(adapter, reader, writer)
}

pub fn run_stdio_with<R, W>(adapter: &McpAdapter, mut reader: R, mut writer: W) -> Result<()>
where
    R: BufRead,
    W: Write,
{
    let mut protocol = McpProtocol::new();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            continue;
        }

        let parsed = match serde_json::from_str::<Value>(trimmed) {
            Ok(parsed) => parsed,
            Err(err) => {
                write_response(&mut writer, parse_error_response(&err.to_string()))?;
                continue;
            }
        };

        if let Some(response) = protocol.handle_message(adapter, parsed) {
            write_response(&mut writer, response)?;
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_response<W>(writer: &mut W, response: Value) -> Result<()>
where
    W: Write,
{
    let encoded = serde_json::to_string(&response)?;
    writer.write_all(encoded.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn parse_error_response(details: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": null,
        "error": {
            "code": CODE_PARSE_ERROR,
            "message": "Parse error",
            "data": details
        }
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use kbolt_core::engine::Engine;
    use serde_json::Value;

    use super::run_stdio_with;
    use crate::protocol::MCP_PROTOCOL_VERSION;
    use crate::test_support::with_isolated_xdg_dirs;
    use crate::McpAdapter;

    fn parse_output_lines(output: &[u8]) -> Vec<Value> {
        let stdout = String::from_utf8_lossy(output);
        stdout
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("line should be valid json"))
            .collect::<Vec<_>>()
    }

    #[test]
    fn emits_parse_error_and_continues() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);
            let input = "{invalid-json\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n";

            let mut output = Vec::new();
            run_stdio_with(&adapter, Cursor::new(input.as_bytes()), &mut output)
                .expect("run stdio");

            let responses = parse_output_lines(&output);
            assert_eq!(responses.len(), 2);
            assert_eq!(responses[0]["error"]["code"], -32700);
            assert_eq!(responses[1]["result"], serde_json::json!({}));
        });
    }

    #[test]
    fn handles_initialize_and_tools_list_over_newline_transport() {
        with_isolated_xdg_dirs(|| {
            let engine = Engine::new(None).expect("create engine");
            let adapter = McpAdapter::new(engine);

            let initialize = format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"{}\",\"capabilities\":{{}},\"clientInfo\":{{\"name\":\"test\",\"version\":\"0.1.0\"}}}}}}\n",
                MCP_PROTOCOL_VERSION
            );
            let initialized = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
            let tools_list =
                "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n";
            let input = format!("{initialize}{initialized}{tools_list}");

            let mut output = Vec::new();
            run_stdio_with(&adapter, Cursor::new(input.as_bytes()), &mut output)
                .expect("run stdio");

            let responses = parse_output_lines(&output);
            assert_eq!(responses.len(), 2);

            assert_eq!(responses[0]["id"], 1);
            assert_eq!(
                responses[0]["result"]["protocolVersion"],
                MCP_PROTOCOL_VERSION
            );

            assert_eq!(responses[1]["id"], 2);
            let tools = responses[1]["result"]["tools"]
                .as_array()
                .expect("tools should be array");
            assert_eq!(tools.len(), 5);
        });
    }
}
