import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, SafeAreaView, Alert, ActivityIndicator } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import LoginScreen from './components/LoginScreen';
import { auth, db } from './firebase'; // Import from your config file
import { onAuthStateChanged, signOut, User } from 'firebase/auth';
import { doc, getDoc, DocumentData, setDoc, collection } from 'firebase/firestore';
import AppNavigator from './navigation/AppNavigator';
import { COLORS } from './constants/theme';
import { OnboardingData } from './components/OnboardingScreen';

export interface UserData {
  uid: string;
  email: string;
  name?: string;
  created?: string;
  settings: {
    height: number;
    weight: number;
    bodyFat: number | null;
    activityLevel: string;
    fitnessGoal: string;
    weeklyWorkouts: number;
    workoutGoal: number;
    calorieGoal: number;
    gender: 'male' | 'female';
    wakeTime: string;
    workoutTime: string;
    sleepTime: string;
    macroSplit: {
      protein: number;
      carbs: number;
      fats: number;
    };
    mealPlan: {
      numberOfMeals: number;
      meals: Array<{
        time: string;
        calories: number;
        macros: {
          protein: number;
          carbs: number;
          fats: number;
        };
      }>;
    };
  };
  workouts?: Array<{
    id: string;
    date: string;
    exercises: Array<{
      name: string;
      sets: number;
      reps: number;
      weight?: number;
    }>;
  }>;
  calorieEntries?: Array<{
    id: string;
    date: string;
    food: string;
    calories: number;
    mealType: string;
  }>;
}

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [userData, setUserData] = useState<UserData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [showOnboarding, setShowOnboarding] = useState<boolean>(false);
  const [newUser, setNewUser] = useState<User | null>(null);

  // Load user data and login state on startup
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user: User | null) => {
      setIsLoading(true);
      try {
        if (user) {
          console.log("App.tsx - User is signed in:", user.uid);
          // Fetch user profile from Firestore
          const userDoc = await getDoc(doc(db, "users", user.uid));
          
          if (userDoc.exists()) {
            const userProfile = userDoc.data() as Omit<UserData, 'uid'>;
            setUserData({
              uid: user.uid,
              ...userProfile
            });
            setIsLoggedIn(true);
            setShowOnboarding(false);
          } else {
            console.log("App.tsx - No user profile found for:", user.uid);
            setNewUser(user);
            setShowOnboarding(true);
          }
        } else {
          console.log("App.tsx - User is signed out");
          setIsLoggedIn(false);
          setUserData(null);
          setShowOnboarding(false);
          setNewUser(null);
        }
      } catch (error) {
        console.log('App.tsx - Error fetching user data:', error);
        setIsLoggedIn(false);
      } finally {
        setIsLoading(false);
      }
    });

    // Clean up subscription
    return () => unsubscribe();
  }, []);

  const handleLogin = (user: UserData) => {
    setUserData(user);
    setIsLoggedIn(true);
    setShowOnboarding(false);
  };

  const calculateMealPlan = (
    calorieGoal: number,
    macroSplit: { protein: number; carbs: number; fats: number },
    wakeTime: string,
    workoutTime: string,
    sleepTime: string,
    numberOfMeals: number
  ) => {
    // Convert protein requirement to grams (4 calories per gram)
    const dailyProteinCalories = (calorieGoal * macroSplit.protein) / 100;
    const dailyProteinGrams = dailyProteinCalories / 4;
    
    // Calculate optimal number of meals based on protein per meal
    // If more than 40g protein per meal needed, add another meal
    if (!numberOfMeals) {
      numberOfMeals = Math.max(4, Math.ceil(dailyProteinGrams / 40));
    }
    
    // Calculate calories and macros per meal
    const caloriesPerMeal = calorieGoal / numberOfMeals;
    const macrosPerMeal = {
      protein: Math.round((calorieGoal * macroSplit.protein / 100) / numberOfMeals / 4), // g
      carbs: Math.round((calorieGoal * macroSplit.carbs / 100) / numberOfMeals / 4), // g
      fats: Math.round((calorieGoal * macroSplit.fats / 100) / numberOfMeals / 9), // g
    };

    // Convert times to minutes since midnight for calculations
    const getMinutes = (time: string) => {
      const [hours, minutes] = time.split(':').map(Number);
      return hours * 60 + minutes;
    };

    const wakeMinutes = getMinutes(wakeTime);
    const workoutMinutes = getMinutes(workoutTime);
    const sleepMinutes = getMinutes(sleepTime);

    // Calculate meal times
    const meals = [];
    const mealGap = 180; // 3 hours between meals
    let currentTime = wakeMinutes;

    for (let i = 0; i < numberOfMeals; i++) {
      // If this meal would be after workout time, delay it by 2 hours after workout
      if (currentTime < workoutMinutes && currentTime + mealGap > workoutMinutes) {
        currentTime = workoutMinutes + 120; // 2 hours after workout
      }

      // Format time back to HH:mm
      const hours = Math.floor(currentTime / 60);
      const minutes = currentTime % 60;
      const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;

      meals.push({
        time: timeString,
        calories: Math.round(caloriesPerMeal),
        macros: macrosPerMeal,
      });

      currentTime += mealGap;
    }

    return {
      numberOfMeals,
      meals,
    };
  };

  const handleOnboardingComplete = async (data: OnboardingData) => {
    try {
      if (!newUser) return;

      // Convert height to cm if in inches
      let heightInCm = parseFloat(data.height);
      if (data.heightUnit === 'in') {
        heightInCm = heightInCm * 2.54;
      }

      // Convert weight to kg if in lbs
      let weightInKg = parseFloat(data.weight);
      if (data.weightUnit === 'lbs') {
        weightInKg = weightInKg * 0.453592;
      }

      // Calculate daily calorie goal using Mifflin-St Jeor Equation
      const bmr = (10 * weightInKg) + (6.25 * heightInCm) - (5 * 30) + (data.gender === 'male' ? 5 : -161);
      
      const activityMultipliers: Record<string, number> = {
        'Sedentary (little or no exercise)': 1.2,
        'Lightly active (light exercise/sports 1-3 days/week)': 1.375,
        'Moderately active (moderate exercise/sports 3-5 days/week)': 1.55,
        'Very active (hard exercise/sports 6-7 days/week)': 1.725,
        'Extra active (very hard exercise/sports & physical job or training twice per day)': 1.9,
      };
      
      const activityMultiplier = activityMultipliers[data.activityLevel] || 1.2;
      const dailyCalorieGoal = Math.round(bmr * activityMultiplier);

      // Calculate meal plan
      const mealPlan = calculateMealPlan(
        dailyCalorieGoal,
        data.macroSplit,
        data.wakeTime,
        data.workoutTime,
        data.sleepTime,
        parseInt(data.mealCount)
      );

      // Create user data object
      const userDataObj: UserData = {
        uid: newUser.uid,
        name: newUser.displayName || newUser.email?.split('@')[0] || 'User',
        email: newUser.email || 'No email',
        created: new Date().toISOString(),
        settings: {
          height: heightInCm,
          weight: weightInKg,
          bodyFat: parseFloat(data.bodyFat) || null,
          activityLevel: data.activityLevel,
          fitnessGoal: data.fitnessGoal,
          weeklyWorkouts: parseInt(data.weeklyWorkouts),
          calorieGoal: dailyCalorieGoal,
          workoutGoal: parseInt(data.weeklyWorkouts),
          gender: data.gender,
          wakeTime: data.wakeTime,
          workoutTime: data.workoutTime,
          sleepTime: data.sleepTime,
          macroSplit: data.macroSplit,
          mealPlan,
        },
        workouts: [],
        calorieEntries: [],
      };

      // Save to Firestore
      await setDoc(doc(db, 'users', newUser.uid), userDataObj);

      // Update local state
      setUserData(userDataObj);
      setIsLoggedIn(true);
      setShowOnboarding(false);
      setNewUser(null);
    } catch (error) {
      console.error('Error saving onboarding data:', error);
      Alert.alert('Error', 'Failed to save your information. Please try again.');
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      // Auth state listener will handle the rest
      console.log("App.tsx - Logout initiated");
    } catch (error) {
      console.log('App.tsx - Error during logout:', error);
      Alert.alert('Error', 'Failed to log out');
    }
  };

  // Show loading state
  if (isLoading) {
    return (
      <SafeAreaView style={styles.loadingContainer}>
        <StatusBar style="auto" />
        <View style={styles.loadingContent}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      </SafeAreaView>
    );
  }

  // Show login screen if not logged in
  if (!isLoggedIn) {
    return (
      <>
        <StatusBar style="auto" />
        <LoginScreen onLogin={handleLogin} />
      </>
    );
  }

  // Show main app with navigation if logged in
  return (
    <>
      <StatusBar style="auto" />
      <AppNavigator 
        userData={userData} 
        onLogout={handleLogout}
        showOnboarding={showOnboarding}
        onOnboardingComplete={handleOnboardingComplete}
        onUpdateUserData={setUserData}
      />
    </>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  loadingContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: COLORS.text,
  },
});