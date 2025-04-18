import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, Image, TextInput, Alert, Modal } from 'react-native';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import { doc, setDoc, updateDoc } from 'firebase/firestore';
import { db } from '../firebase';
import { UserData } from '../App'; // Import the UserData type
import { Picker } from '@react-native-picker/picker';

interface ProfileScreenProps {
  userData: UserData;
  onLogout: () => void;
  onUpdateUserData: (userData: UserData) => void; // Add this prop
}

interface MealPlanMeal {
  time: string;
  calories: number;
  macros: {
    protein: number;
    carbs: number;
    fats: number;
  };
}

const ACTIVITY_LEVELS = [
  'Sedentary (little or no exercise)',
  'Lightly active (light exercise/sports 1-3 days/week)',
  'Moderately active (moderate exercise/sports 3-5 days/week)',
  'Very active (hard exercise/sports 6-7 days/week)',
  'Extra active (very hard exercise/sports & physical job or training twice per day)',
];

const FITNESS_GOALS = [
  'Lose Weight',
  'Build Muscle',
  'Improve Strength',
  'Increase Endurance',
  'Maintain Fitness',
];

const ProfileScreen = ({ userData, onLogout, onUpdateUserData }: ProfileScreenProps) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedData, setEditedData] = useState({
    height: userData?.settings?.height?.toString() || '',
    weight: userData?.settings?.weight?.toString() || '',
    bodyFat: userData?.settings?.bodyFat?.toString() || '',
    activityLevel: userData?.settings?.activityLevel || '',
    fitnessGoal: userData?.settings?.fitnessGoal || '',
    weeklyWorkouts: userData?.settings?.weeklyWorkouts?.toString() || '',
    heightUnit: 'cm',
    weightUnit: 'kg',
    wakeTime: userData?.settings?.wakeTime || '06:00',
    workoutTime: userData?.settings?.workoutTime || '17:00',
    sleepTime: userData?.settings?.sleepTime || '22:00',
    macroSplit: {
      protein: userData?.settings?.macroSplit?.protein || 30,
      carbs: userData?.settings?.macroSplit?.carbs || 35,
      fats: userData?.settings?.macroSplit?.fats || 35,
    },
  });

  const [isEditingMealCount, setIsEditingMealCount] = useState(false);
  const [mealCount, setMealCount] = useState(userData.settings.mealPlan.numberOfMeals.toString());

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
    
    // Ensure last meal is at least 3 hours before bedtime
    const lastMealCutoff = sleepMinutes - 120; // 3 hours before sleep
    const availableMinutes = lastMealCutoff - wakeMinutes;
    
    // Determine how many meals we can fit
    const actualMealCount = Math.min(numberOfMeals, Math.floor(availableMinutes / 120) + 1);
    const mealGap = Math.floor(availableMinutes / (actualMealCount));
    
    let currentTime = wakeMinutes;
    
    // Keep adding meals until we reach the actual meal count
    for (let i = 0; i < actualMealCount; i++) {
      // First meal is at wake time, then space them out
      if (i === 0) {
        currentTime = wakeMinutes + 30; // 30 mins after waking up
      } else {
        currentTime += mealGap;
      }
      
      // If this meal would conflict with workout time (within 1 hour before/after), adjust it
      if (Math.abs(currentTime - workoutMinutes) < 60) {
        if (currentTime < workoutMinutes) {
          currentTime = workoutMinutes - 70; // Just before workout window
        } else {
          currentTime = workoutMinutes + 70; // Just after workout window
        }
      }

      // Format time back to HH:mm AM/PM format
      const date = new Date();
      date.setHours(Math.floor(currentTime / 60));
      date.setMinutes(currentTime % 60);
      const timeString = date.toLocaleTimeString([], { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
      });

      meals.push({
        time: timeString,
        calories: Math.round(calorieGoal / actualMealCount),
        macros: {
          protein: Math.round((calorieGoal * macroSplit.protein / 100) / actualMealCount / 4),
          carbs: Math.round((calorieGoal * macroSplit.carbs / 100) / actualMealCount / 4),
          fats: Math.round((calorieGoal * macroSplit.fats / 100) / actualMealCount / 9),
        }
      });
    }

    return {
      numberOfMeals: numberOfMeals,
      meals,
    };
  };

  const handleSave = async () => {
    try {
      // Convert height to cm if in inches
      let heightInCm = parseFloat(editedData.height);
      if (editedData.heightUnit === 'in') {
        heightInCm = heightInCm * 2.54;
      }

      // Convert weight to kg if in lbs
      let weightInKg = parseFloat(editedData.weight);
      if (editedData.weightUnit === 'lbs') {
        weightInKg = weightInKg * 0.453592;
      }

      // Calculate daily calorie goal using Mifflin-St Jeor Equation
      const bmr = (10 * weightInKg) + (6.25 * heightInCm) - (5 * 30) + (userData.settings.gender === 'male' ? 5 : -161);
      
      const activityMultipliers: Record<string, number> = {
        'Sedentary (little or no exercise)': 1.2,
        'Lightly active (light exercise/sports 1-3 days/week)': 1.375,
        'Moderately active (moderate exercise/sports 3-5 days/week)': 1.55,
        'Very active (hard exercise/sports 6-7 days/week)': 1.725,
        'Extra active (very hard exercise/sports & physical job or training twice per day)': 1.9,
      };
      
      const activityMultiplier = activityMultipliers[editedData.activityLevel] || 1.2;
      const dailyCalorieGoal = Math.round(bmr * activityMultiplier);

      // Calculate new meal plan
      const mealPlan = calculateMealPlan(
        dailyCalorieGoal,
        editedData.macroSplit,
        editedData.wakeTime,
        editedData.workoutTime,
        editedData.sleepTime,
        parseInt(mealCount)
      );

      const updatedSettings = {
        ...userData.settings,
        height: heightInCm,
        weight: weightInKg,
        bodyFat: parseFloat(editedData.bodyFat) || null,
        activityLevel: editedData.activityLevel,
        fitnessGoal: editedData.fitnessGoal,
        weeklyWorkouts: parseInt(editedData.weeklyWorkouts),
        calorieGoal: dailyCalorieGoal,
        wakeTime: editedData.wakeTime,
        workoutTime: editedData.workoutTime,
        sleepTime: editedData.sleepTime,
        macroSplit: editedData.macroSplit,
        mealPlan,
      };

      const updatedUserData = {
        ...userData,
        settings: updatedSettings,
      };

      // Save to Firestore
      await setDoc(doc(db, 'users', userData.uid), updatedUserData, { merge: true });

      // Update local state through the parent component
      onUpdateUserData(updatedUserData);
      setIsEditing(false);
      Alert.alert('Success', 'Profile updated successfully');
    } catch (error) {
      console.error('Error updating profile:', error);
      Alert.alert('Error', 'Failed to update profile');
    }
  };

  const handleUpdateMealCount = async () => {
    try {
      const parsedMealCount = parseInt(mealCount);
      if (isNaN(parsedMealCount) || parsedMealCount < 2 || parsedMealCount > 8) {
        Alert.alert('Invalid Input', 'Please enter a number between 2 and 8');
        return;
      }

      // Create new meal plan with updated meal count
      const mealPlan = calculateMealPlan(
        userData.settings.calorieGoal,
        userData.settings.macroSplit,
        userData.settings.wakeTime,
        userData.settings.workoutTime,
        userData.settings.sleepTime,
        parsedMealCount
      );

      const updatedUserData = {
        ...userData,
        settings: {
          ...userData.settings,
          mealPlan,
        }
      };

      await updateDoc(doc(db, 'users', userData.uid), {
        settings: updatedUserData.settings
      });

      onUpdateUserData(updatedUserData);
      setIsEditingMealCount(false);
      Alert.alert('Success', 'Meal plan updated successfully');
    } catch (error) {
      console.error('Error updating meal count:', error);
      Alert.alert('Error', 'Failed to update meal count');
    }
  };

  const renderEditableStats = () => (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>Edit Your Stats</Text>
      
      <View style={styles.inputContainer}>
        <Text style={styles.label}>Height</Text>
        <View style={styles.unitContainer}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            value={editedData.height}
            onChangeText={(text) => setEditedData({ ...editedData, height: text })}
            keyboardType="numeric"
            placeholder={editedData.heightUnit === 'in' ? "71" : "180"}
          />
          <TouchableOpacity
            style={styles.unitButton}
            onPress={() => setEditedData({ 
              ...editedData, 
              heightUnit: editedData.heightUnit === 'in' ? 'cm' : 'in',
              height: '' // Clear the value when switching units
            })}
          >
            <Text style={styles.unitButtonText}>{editedData.heightUnit}</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Weight</Text>
        <View style={styles.unitContainer}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            value={editedData.weight}
            onChangeText={(text) => setEditedData({ ...editedData, weight: text })}
            keyboardType="numeric"
            placeholder={editedData.weightUnit === 'lbs' ? "180" : "82"}
          />
          <TouchableOpacity
            style={styles.unitButton}
            onPress={() => setEditedData({ 
              ...editedData, 
              weightUnit: editedData.weightUnit === 'lbs' ? 'kg' : 'lbs',
              weight: '' // Clear the value when switching units
            })}
          >
            <Text style={styles.unitButtonText}>{editedData.weightUnit}</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Body Fat % (optional)</Text>
        <TextInput
          style={styles.input}
          value={editedData.bodyFat}
          onChangeText={(text) => setEditedData({ ...editedData, bodyFat: text })}
          keyboardType="numeric"
          placeholder="Enter your body fat percentage"
        />
      </View>

      <Text style={styles.label}>Activity Level</Text>
      {ACTIVITY_LEVELS.map((level, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.optionButton,
            editedData.activityLevel === level && styles.selectedOption,
          ]}
          onPress={() => setEditedData({ ...editedData, activityLevel: level })}
        >
          <Text style={[
            styles.optionText,
            editedData.activityLevel === level && styles.selectedOptionText,
          ]}>
            {level}
          </Text>
        </TouchableOpacity>
      ))}

      <Text style={[styles.label, { marginTop: 20 }]}>Fitness Goal</Text>
      {FITNESS_GOALS.map((goal, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.optionButton,
            editedData.fitnessGoal === goal && styles.selectedOption,
          ]}
          onPress={() => setEditedData({ ...editedData, fitnessGoal: goal })}
        >
          <Text style={[
            styles.optionText,
            editedData.fitnessGoal === goal && styles.selectedOptionText,
          ]}>
            {goal}
          </Text>
        </TouchableOpacity>
      ))}

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Weekly Workouts Target</Text>
        <TextInput
          style={styles.input}
          value={editedData.weeklyWorkouts}
          onChangeText={(text) => setEditedData({ ...editedData, weeklyWorkouts: text })}
          keyboardType="numeric"
          placeholder="Enter number of workouts"
        />
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Daily Schedule</Text>
        <View style={styles.timeContainer}>
          <View style={styles.timeInput}>
            <Text style={styles.timeLabel}>Wake Up</Text>
            <TextInput
              style={styles.input}
              value={editedData.wakeTime}
              onChangeText={(text) => setEditedData({ ...editedData, wakeTime: text })}
              placeholder="06:00"
              keyboardType="numbers-and-punctuation"
            />
          </View>
          <View style={styles.timeInput}>
            <Text style={styles.timeLabel}>Workout</Text>
            <TextInput
              style={styles.input}
              value={editedData.workoutTime}
              onChangeText={(text) => setEditedData({ ...editedData, workoutTime: text })}
              placeholder="17:00"
              keyboardType="numbers-and-punctuation"
            />
          </View>
          <View style={styles.timeInput}>
            <Text style={styles.timeLabel}>Sleep</Text>
            <TextInput
              style={styles.input}
              value={editedData.sleepTime}
              onChangeText={(text) => setEditedData({ ...editedData, sleepTime: text })}
              placeholder="22:00"
              keyboardType="numbers-and-punctuation"
            />
          </View>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Macro Split (%)</Text>
        <View style={styles.macroContainer}>
          <View style={styles.macroInput}>
            <Text style={styles.macroLabel}>Protein</Text>
            <TextInput
              style={[styles.input, styles.macroValue]}
              value={editedData.macroSplit.protein.toString()}
              onChangeText={(text) => setEditedData({
                ...editedData,
                macroSplit: {
                  ...editedData.macroSplit,
                  protein: parseInt(text) || 0
                }
              })}
              keyboardType="numeric"
              placeholder="30"
            />
          </View>
          <View style={styles.macroInput}>
            <Text style={styles.macroLabel}>Carbs</Text>
            <TextInput
              style={[styles.input, styles.macroValue]}
              value={editedData.macroSplit.carbs.toString()}
              onChangeText={(text) => setEditedData({
                ...editedData,
                macroSplit: {
                  ...editedData.macroSplit,
                  carbs: parseInt(text) || 0
                }
              })}
              keyboardType="numeric"
              placeholder="35"
            />
          </View>
          <View style={styles.macroInput}>
            <Text style={styles.macroLabel}>Fats</Text>
            <TextInput
              style={[styles.input, styles.macroValue]}
              value={editedData.macroSplit.fats.toString()}
              onChangeText={(text) => setEditedData({
                ...editedData,
                macroSplit: {
                  ...editedData.macroSplit,
                  fats: parseInt(text) || 0
                }
              })}
              keyboardType="numeric"
              placeholder="35"
            />
          </View>
        </View>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Number of Meals</Text>
        <Text style={styles.settingValue}>{mealCount} meals per day</Text>
      </View>

      <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
        <Text style={styles.saveButtonText}>Save Changes</Text>
      </TouchableOpacity>
    </View>
  );

  const renderViewStats = () => (
    <>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Your Stats</Text>
        <View style={styles.goalItem}>
          <Text style={styles.goalLabel}>Height</Text>
          <Text style={styles.goalValue}>{userData?.settings?.height ? `${Math.round(userData.settings.height)} cm` : 'Not set'}</Text>
        </View>
        <View style={styles.goalItem}>
          <Text style={styles.goalLabel}>Weight</Text>
          <Text style={styles.goalValue}>{userData?.settings?.weight ? `${Math.round(userData.settings.weight)} kg` : 'Not set'}</Text>
        </View>
        <View style={styles.goalItem}>
          <Text style={styles.goalLabel}>Body Fat</Text>
          <Text style={styles.goalValue}>{userData?.settings?.bodyFat ? `${userData.settings.bodyFat}%` : 'Not set'}</Text>
        </View>
        <TouchableOpacity style={styles.editButton} onPress={() => setIsEditing(true)}>
          <Text style={styles.editButtonText}>Edit Stats</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Fitness Goals</Text>
        <View style={styles.goalItem}>
          <Text style={styles.goalLabel}>Primary Goal</Text>
          <Text style={styles.goalValue}>{userData?.settings?.fitnessGoal || 'Not set'}</Text>
        </View>
        <View style={styles.goalItem}>
          <Text style={styles.goalLabel}>Daily Calorie Target</Text>
          <Text style={styles.goalValue}>{userData?.settings?.calorieGoal || 2200} calories</Text>
        </View>
        <View style={styles.goalItem}>
          <Text style={styles.goalLabel}>Weekly Workout Target</Text>
          <Text style={styles.goalValue}>{userData?.settings?.weeklyWorkouts || 5} workouts</Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Meal Plan</Text>
        {userData?.settings?.mealPlan?.meals.map((meal: MealPlanMeal, index: number) => (
          <View key={index} style={styles.mealItem}>
            <Text style={styles.mealTime}>{meal.time}</Text>
            <Text style={styles.mealCalories}>{meal.calories} cal</Text>
            <Text style={styles.mealMacros}>
              P: {meal.macros.protein}g | C: {meal.macros.carbs}g | F: {meal.macros.fats}g
            </Text>
          </View>
        ))}
      </View>
    </>
  );

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Image 
          source={{ uri: 'https://placehold.co/200x200/9b59b6/FFFFFF?text=🏋️' }} 
          style={styles.profileImage} 
        />
        <Text style={styles.userName}>{userData?.name || 'User'}</Text>
        <Text style={styles.userEmail}>{userData?.email || 'No email'}</Text>
      </View>

      {isEditing ? renderEditableStats() : renderViewStats()}
      
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Stats Overview</Text>
        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{userData?.workouts?.length || 0}</Text>
            <Text style={styles.statLabel}>Workouts</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{userData?.calorieEntries?.length || 0}</Text>
            <Text style={styles.statLabel}>Meals</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>0</Text>
            <Text style={styles.statLabel}>Achievements</Text>
          </View>
        </View>
      </View>
      
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Account Information</Text>
        <Text style={styles.infoItem}>Member since: {
          userData?.created 
            ? new Date(userData.created).toLocaleDateString() 
            : 'Unknown'
        }</Text>
      </View>
      
      <TouchableOpacity style={styles.logoutButton} onPress={onLogout}>
        <Text style={styles.logoutText}>Logout</Text>
      </TouchableOpacity>

      {/* Meal Count Modal */}
      <Modal
        visible={isEditingMealCount}
        animationType="slide"
        transparent={true}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Number of Meals</Text>
            <Text style={styles.modalInstructions}>
              Choose how many meals you want per day (2-8)
            </Text>
            
            <Picker
              selectedValue={mealCount}
              onValueChange={(itemValue) => setMealCount(itemValue)}
              style={styles.picker}
            >
              {[2, 3, 4, 5, 6, 7, 8].map((num) => (
                <Picker.Item key={num} label={`${num} meals`} value={num.toString()} />
              ))}
            </Picker>
            
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => {
                  setIsEditingMealCount(false);
                  setMealCount(userData.settings.mealPlan.numberOfMeals.toString());
                }}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[styles.modalButton, styles.saveButton]}
                onPress={handleUpdateMealCount}
              >
                <Text style={styles.saveButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  header: {
    backgroundColor: COLORS.primary,
    padding: 30,
    alignItems: 'center',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  profileImage: {
    width: 100,
    height: 100,
    borderRadius: 50,
    borderWidth: 4,
    borderColor: COLORS.card,
    marginBottom: 15,
  },
  userName: {
    fontSize: SIZES.xLarge,
    fontWeight: 'bold',
    color: COLORS.card,
    marginBottom: 5,
  },
  userEmail: {
    fontSize: SIZES.medium,
    color: COLORS.card,
    opacity: 0.8,
  },
  card: {
    backgroundColor: COLORS.card,
    margin: 15,
    borderRadius: 15,
    padding: 20,
    ...SHADOWS.small,
  },
  cardTitle: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 15,
  },
  goalItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  goalLabel: {
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  goalValue: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  editButton: {
    backgroundColor: COLORS.accent,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 15,
  },
  editButtonText: {
    color: COLORS.card,
    fontWeight: 'bold',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statBox: {
    flex: 1,
    alignItems: 'center',
    padding: 15,
  },
  statValue: {
    fontSize: SIZES.xLarge,
    fontWeight: 'bold',
    color: COLORS.secondary,
    marginBottom: 5,
  },
  statLabel: {
    fontSize: SIZES.medium,
    color: COLORS.subtext,
  },
  infoItem: {
    fontSize: SIZES.medium,
    color: COLORS.text,
    paddingVertical: 8,
  },
  logoutButton: {
    backgroundColor: COLORS.error,
    margin: 15,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 30,
    ...SHADOWS.small,
  },
  logoutText: {
    color: COLORS.card,
    fontWeight: 'bold',
    fontSize: SIZES.medium,
  },
  inputContainer: {
    marginBottom: 20,
  },
  label: {
    fontSize: SIZES.medium,
    color: COLORS.text,
    marginBottom: 8,
  },
  input: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    fontSize: SIZES.medium,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  unitContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  unitButton: {
    backgroundColor: COLORS.accent,
    padding: 15,
    borderRadius: 10,
    minWidth: 60,
    alignItems: 'center',
  },
  unitButtonText: {
    color: COLORS.card,
    fontWeight: 'bold',
    fontSize: SIZES.medium,
  },
  optionButton: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  selectedOption: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  optionText: {
    fontSize: SIZES.medium,
    color: COLORS.text,
  },
  selectedOptionText: {
    color: COLORS.card,
  },
  saveButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    marginTop: 20,
  },
  saveButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  timeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  timeInput: {
    flex: 1,
  },
  timeLabel: {
    fontSize: SIZES.small,
    color: COLORS.text,
    marginBottom: 4,
  },
  macroContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  macroInput: {
    flex: 1,
  },
  macroLabel: {
    fontSize: SIZES.small,
    color: COLORS.text,
    marginBottom: 4,
  },
  macroValue: {
    textAlign: 'center',
  },
  mealItem: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    ...SHADOWS.small,
  },
  mealTime: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 5,
  },
  mealCalories: {
    fontSize: SIZES.medium,
    color: COLORS.primary,
    marginBottom: 5,
  },
  mealMacros: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
  },
  picker: {
    width: '100%',
    marginBottom: 20,
  },
  modalInstructions: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginBottom: 20,
    textAlign: 'center',
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    backgroundColor: COLORS.card,
    padding: 20,
    borderRadius: 10,
    width: '80%',
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 10,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
  },
  cancelButton: {
    backgroundColor: COLORS.error,
    padding: 15,
    borderRadius: 8,
    flex: 1,
    marginRight: 5,
  },
  cancelButtonText: {
    color: COLORS.card,
    fontWeight: 'bold',
    fontSize: SIZES.medium,
  },
  modalButton: {
    backgroundColor: COLORS.primary,
    padding: 15,
    borderRadius: 8,
    flex: 1,
    marginLeft: 5,
  },
  modalButtonText: {
    color: COLORS.card,
    fontWeight: 'bold',
    fontSize: SIZES.medium,
  },
  settingValue: {
    fontSize: SIZES.medium,
    color: COLORS.primary,
    fontWeight: 'bold',
  },
});

export default ProfileScreen; 