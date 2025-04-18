import { initializeApp } from 'firebase/app';
import { initializeAuth } from 'firebase/auth';
import * as firebaseAuth from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import AsyncStorage from '@react-native-async-storage/async-storage';
// Note: getAnalytics doesn't work directly with Expo unless you use EAS or custom native code
// import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "you cant see this",
  authDomain: "secret",
  projectId: "sceret",
  storageBucket: "secret",
  messagingSenderId: "secret",
  appId: "secret",
  measurementId: "secret"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
// Analytics doesn't work directly in Expo managed workflow
// const analytics = getAnalytics(app);

// Initialize Auth with AsyncStorage persistence
const auth = initializeAuth(app, {
  persistence: (firebaseAuth as any).getReactNativePersistence(AsyncStorage)
});

const db = getFirestore(app);

export { auth, db }; 