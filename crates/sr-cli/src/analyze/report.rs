use super::Report;

/// Resolve aliases before writing either output, including not-yet-created
/// files. Creating output parents also catches invalid destinations early.
pub(super) fn output_identity(path: &Path) -> Result<String> {
    let name = path.file_name().context("output path needs a filename")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let resolved = if path.exists() {
        path.canonicalize()?
    } else {
        parent.canonicalize()?.join(name)
    };
    let identity = resolved.to_string_lossy().into_owned();
    Ok(if cfg!(windows) {
        identity.to_lowercase()
    } else {
        identity
    })
}
use anyhow::{Context, Result};
use std::path::Path;

fn embedded_json<T: serde::Serialize>(data: &T) -> Result<String> {
    // A JSON script is still terminated by </script> in an HTML parser.
    Ok(serde_json::to_string(data)?
        .replace('<', "\\u003c")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029"))
}

pub(super) fn html(report: &Report) -> Result<String> {
    Ok(include_str!("report.html")
        .replace("__REPORT_JSON__", &embedded_json(report)?)
        .replace("__INSIGHTS_JS__", include_str!("insights.js"))
        .replace("__REVIEW_JS__", include_str!("review.js")))
}

pub(super) fn write(report: &Report, json: &Path, page: &Path) -> Result<()> {
    for p in [json, page] {
        if let Some(parent) = p.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(json, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("writing {}", json.display()))?;
    std::fs::write(page, html(report)?).with_context(|| format!("writing {}", page.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_aliases_have_one_identity() {
        assert_eq!(
            output_identity(Path::new("project-analysis.json")).unwrap(),
            output_identity(Path::new("./project-analysis.json")).unwrap()
        );
    }
    #[test]
    fn embedded_data_cannot_close_script_or_execute_markup() {
        let source = serde_json::json!({"file":"</script><img src=x onerror=alert(1)>&\u{2028}"});
        let encoded = embedded_json(&source).unwrap();
        assert!(!encoded.contains('<'));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&encoded).unwrap(),
            source
        );
    }
    #[test]
    fn report_is_offline_and_uses_rust_results() {
        let template = include_str!("report.html");
        for forbidden in [
            "https://",
            "http://",
            "fetch(",
            "XMLHttpRequest",
            "<script src=",
            "<link ",
        ] {
            assert!(!template.contains(forbidden), "{forbidden}");
        }
        assert!(template.contains("local_fit"));
        assert!(template.contains("ideal_noise"));
        assert!(!template.contains("Math.pow"));
    }
}
