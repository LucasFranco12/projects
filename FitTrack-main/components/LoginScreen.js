import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  TouchableOpacity, 
  KeyboardAvoidingView, 
  Platform,
  Alert,
  ActivityIndicator
} from 'react-native';
import { auth } from '../firebase';
import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword 
} from 'firebase/auth';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import LogoImage from './LogoImage';

const LoginScreen = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [name, setName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAuth = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    if (isRegistering && !name) {
      Alert.alert('Error', 'Please enter your name');
      return;
    }

    setIsLoading(true);
    
    try {
      let user;
      if (isRegistering) {
        console.log('Starting registration process...');
        // Register new user with Firebase
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        user = userCredential.user;
        console.log('User registered successfully:', user.uid);
      } else {
        console.log('Starting login process...');
        // Sign in existing user with Firebase
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        user = userCredential.user;
        console.log('User logged in successfully:', user.uid);
      }

      // Pass user data to parent component
      onLogin({
        uid: user.uid,
        email: user.email,
        name: name || user.displayName || 'User'
      });
    } catch (error) {
      console.log('Error during authentication:', error);
      
      // Show more specific error messages
      if (error.code === 'auth/email-already-in-use') {
        Alert.alert('Error', 'This email is already registered');
      } else if (error.code === 'auth/invalid-email') {
        Alert.alert('Error', 'Invalid email address');
      } else if (error.code === 'auth/weak-password') {
        Alert.alert('Error', 'Password should be at least 6 characters');
      } else if (error.code === 'auth/user-not-found' || error.code === 'auth/wrong-password') {
        Alert.alert('Error', 'Invalid email or password');
      } else {
        Alert.alert('Error', 'Authentication failed');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // For easy testing - remove in production
  const useTestCredentials = () => {
    if (isRegistering) {
      setName('Test User');
      setEmail('test@example.com');
      setPassword('password123');
    } else {
      setEmail('test@example.com');
      setPassword('password123');
    }
  };

  return (
    <KeyboardAvoidingView 
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={styles.container}
    >
      <View style={styles.logoContainer}>
        <LogoImage size={200} withText={false} />
        <Text style={styles.logoText}>FitTrack</Text>
        <Text style={styles.tagline}>Your journey to a healthier you</Text>
      </View>

      <View style={styles.formContainer}>
        {isRegistering && (
          <TextInput
            style={styles.input}
            placeholder="Your Name"
            value={name}
            onChangeText={setName}
            autoCapitalize="words"
          />
        )}

        <TextInput
          style={styles.input}
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />

        <TextInput
          style={styles.input}
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        {error && <Text style={styles.error}>{error}</Text>}

        <TouchableOpacity 
          style={[styles.button, isLoading && styles.disabledButton]} 
          onPress={handleAuth}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.buttonText}>
              {isRegistering ? 'Create Account' : 'Login'}
            </Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.switchButton} 
          onPress={() => {
            setIsRegistering(!isRegistering);
            setEmail('');
            setPassword('');
            setName('');
          }}
          disabled={isLoading}
        >
          <Text style={styles.switchButtonText}>
            {isRegistering 
              ? 'Already have an account? Login' 
              : 'New user? Create account'}
          </Text>
        </TouchableOpacity>
        
        {/* Debug helper - remove in production */}
        <TouchableOpacity 
          style={styles.testButton} 
          onPress={useTestCredentials}
          disabled={isLoading}
        >
          <Text style={styles.testButtonText}>Fill Test Credentials</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
    justifyContent: 'center',
    padding: 20,
  },
  logoContainer: {
    alignItems: 'center',
    marginBottom: 40,
  },
  logoText: {
    fontSize: SIZES.xxLarge,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginTop: 20,
    marginBottom: 10,
  },
  tagline: {
    fontSize: SIZES.medium,
    color: COLORS.subtext,
  },
  formContainer: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 20,
    ...SHADOWS.medium,
  },
  input: {
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 15,
    borderRadius: 5,
    marginBottom: 15,
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  button: {
    backgroundColor: COLORS.primary,
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
  },
  buttonText: {
    color: COLORS.card,
    fontWeight: 'bold',
    fontSize: SIZES.medium,
  },
  switchButton: {
    marginTop: 20,
    alignItems: 'center',
  },
  switchButtonText: {
    color: COLORS.primary,
    fontSize: SIZES.medium,
  },
  testButton: {
    marginTop: 15,
    padding: 8,
    backgroundColor: COLORS.border,
    borderRadius: 5,
    alignItems: 'center',
  },
  testButtonText: {
    color: COLORS.subtext,
    fontSize: SIZES.small,
  },
  disabledButton: {
    backgroundColor: COLORS.accent,
    opacity: 0.7,
  },
  error: {
    color: COLORS.error,
    marginBottom: 15,
    textAlign: 'center',
  },
});

export default LoginScreen;