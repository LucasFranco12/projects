use anyhow::{Result, anyhow};
use serde::Deserialize;

pub struct AuthTokens {
    pub id_token: String,
    pub refresh_token: String,
    pub expires_at: i64,
    pub local_id: String,  // Firebase user ID
}

#[derive(Deserialize)]
struct FirebaseLoginResponse {
    idToken: String,
    refreshToken: String,
    expiresIn: String,
    localId: String,
}

#[derive(Deserialize)]
struct FirebaseSignupResponse {
    idToken: String,
    refreshToken: String,
    expiresIn: String,
    localId: String,
}

pub fn create_account(email: &str, password: &str, api_key: &str) -> Result<AuthTokens> {
    let url = format!(
        "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={}",
        api_key
    );
    let payload = serde_json::json!({
        "email": email,
        "password": password,
        "returnSecureToken": true
    });
    let client = reqwest::blocking::Client::new();
    let resp = client.post(&url).json(&payload).send()?;
    
    if !resp.status().is_success() {
        let err_text = resp.text().unwrap_or_default();
        return Err(anyhow!("Firebase account creation failed: {}", err_text));
    }
    
    let signup_resp: FirebaseSignupResponse = resp.json()?;
    let expires_in: i64 = signup_resp.expiresIn.parse().unwrap_or(3600);
    let expires_at = chrono::Utc::now().timestamp() + expires_in;
    
    Ok(AuthTokens {
        id_token: signup_resp.idToken,
        refresh_token: signup_resp.refreshToken,
        expires_at,
        local_id: signup_resp.localId,
    })
}

pub fn login(email: &str, password: &str, api_key: &str) -> Result<AuthTokens> {
    let url = format!(
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={}",
        api_key
    );
    let payload = serde_json::json!({
        "email": email,
        "password": password,
        "returnSecureToken": true
    });
    let client = reqwest::blocking::Client::new();
    let resp = client.post(&url).json(&payload).send()?;
    
    if !resp.status().is_success() {
        let err_text = resp.text().unwrap_or_default();
        return Err(anyhow!("Firebase login failed: {}", err_text));
    }
    
    let login_resp: FirebaseLoginResponse = resp.json()?;
    let expires_in: i64 = login_resp.expiresIn.parse().unwrap_or(3600);
    let expires_at = chrono::Utc::now().timestamp() + expires_in;
    
    Ok(AuthTokens {
        id_token: login_resp.idToken,
        refresh_token: login_resp.refreshToken,
        expires_at,
        local_id: login_resp.localId,
    })
}

pub fn refresh_token(refresh_token: &str, api_key: &str) -> Result<AuthTokens> {
    let url = format!(
        "https://securetoken.googleapis.com/v1/token?key={}",
        api_key
    );
    let payload = serde_json::json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    });
    let client = reqwest::blocking::Client::new();
    let resp = client.post(&url).json(&payload).send()?;
    
    if !resp.status().is_success() {
        let err_text = resp.text().unwrap_or_default();
        return Err(anyhow!("Firebase token refresh failed: {}", err_text));
    }
    
    #[derive(Deserialize)]
    struct RefreshResponse {
        id_token: String,
        refresh_token: String,
        expires_in: String,
        user_id: String,
    }
    
    let refresh_resp: RefreshResponse = resp.json()?;
    let expires_in: i64 = refresh_resp.expires_in.parse().unwrap_or(3600);
    let expires_at = chrono::Utc::now().timestamp() + expires_in;
    
    Ok(AuthTokens {
        id_token: refresh_resp.id_token,
        refresh_token: refresh_resp.refresh_token,
        expires_at,
        local_id: refresh_resp.user_id,
    })
}
