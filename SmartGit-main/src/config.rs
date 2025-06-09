use serde::{Serialize, Deserialize};
use std::fs;
use std::path::PathBuf;
use crate::auth;

#[derive(Serialize, Deserialize, Default)]
pub struct FirebaseConfig {
    pub api_key: String,
    pub project_id: String,
    pub id_token: String,
    pub refresh_token: String,
    pub expires_at: i64,
    pub project_name: Option<String>, // New: project name for Firestore
    pub user_id: String,  // Store the Firebase user ID
}

impl FirebaseConfig {
    pub fn path() -> PathBuf {
        // Store config in .smartgit/firebase.json in the current working directory
        std::env::current_dir().unwrap().join(".smartgit/firebase.json")
    }
    pub fn load() -> Option<Self> {
        let path = Self::path();
        if path.exists() {
            let data = fs::read_to_string(path).ok()?;
            serde_json::from_str(&data).ok()
        } else {
            None
        }
    }
    pub fn save(&self) {
        let path = Self::path();
        let _ = fs::create_dir_all(path.parent().unwrap());
        let _ = fs::write(path, serde_json::to_string_pretty(self).unwrap());
    }
    // Prompt for project name if not set
    pub fn ensure_project_name(&mut self) {
        if self.project_name.is_none() {
            println!("Enter a name for your project (used as the directory in the database):");
            let mut name = String::new();
            std::io::stdin().read_line(&mut name).expect("Failed to read project name");
            let name = name.trim();
            if !name.is_empty() {
                // Sanitize: replace spaces and forbidden chars with _
                let sanitized = name.replace(|c: char| c == '/' || c == '.' || c == '#' || c == '$' || c == '[' || c == ']' || c.is_whitespace(), "_");
                self.project_name = Some(sanitized);
                self.save();
            }
        }
    }

    // New method to ensure token is valid
    pub fn ensure_valid_token(&mut self) -> anyhow::Result<()> {
        let now = chrono::Utc::now().timestamp();
        
        // If token is expired or will expire in the next 5 minutes
        if now >= self.expires_at - 300 {
            println!("Token expired or expiring soon, refreshing...");
            match auth::refresh_token(&self.refresh_token, &self.api_key) {
                Ok(tokens) => {
                    self.id_token = tokens.id_token;
                    self.refresh_token = tokens.refresh_token;
                    self.expires_at = tokens.expires_at;
                    self.user_id = tokens.local_id;  // Update user ID from refresh
                    self.save();
                    println!("Token refreshed successfully.");
                }
                Err(e) => {
                    eprintln!("Failed to refresh token: {}", e);
                    eprintln!("Please run 'smartgit login' again.");
                    return Err(anyhow::anyhow!("Token refresh failed"));
                }
            }
        }
        Ok(())
    }
}
