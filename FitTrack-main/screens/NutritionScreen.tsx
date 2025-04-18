import React, { useState, useMemo, useEffect } from 'react';
import { StyleSheet, Text, View, FlatList, TouchableOpacity, Modal, TextInput, ScrollView, Alert, ActivityIndicator } from 'react-native';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import { UserData } from '../App';
import { doc, updateDoc } from 'firebase/firestore';
import { db } from '../firebase';
import { Ionicons } from '@expo/vector-icons';
import { searchFoods, USDAFoodItem } from '../utils/foodDatabase';
import AsyncStorage from '@react-native-async-storage/async-storage';
import FoodItem from '../components/FoodItem';
import LogoImage from '../components/LogoImage';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import DateSelector from '../components/DateSelector';
import MealCard from '../components/MealCard';

interface MealEntry {
  id: string;
  date: string;
  food: string;
  calories: number;
  mealType: string;
  macros: {
    protein: number;
    carbs: number;
    fats: number;
  };
}

interface NutritionScreenProps {
  userData: UserData;
  onUpdateUserData: (userData: UserData) => void;
}

interface RecentFood extends USDAFoodItem {
  lastUsed: string;
}

interface FavoriteFood extends USDAFoodItem {
  dateAdded: string;
}

const NutritionScreen = ({ userData, onUpdateUserData }: NutritionScreenProps) => {
  const [showAddMealModal, setShowAddMealModal] = useState(false);
  const [showEditMealModal, setShowEditMealModal] = useState(false);
  const [selectedMealTime, setSelectedMealTime] = useState<string>('');
  const [selectedMeal, setSelectedMeal] = useState<MealEntry | null>(null);
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [newMeal, setNewMeal] = useState({
    food: '',
    protein: '',
    carbs: '',
    fats: '',
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<USDAFoodItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [recentFoods, setRecentFoods] = useState<USDAFoodItem[]>([]);
  const [favoriteFoods, setFavoriteFoods] = useState<USDAFoodItem[]>([]);
  const [servingMultiplier, setServingMultiplier] = useState(1);
  const [customFoods, setCustomFoods] = useState<USDAFoodItem[]>([]);
  const [showFoodSearchModal, setShowFoodSearchModal] = useState(false);
  const [showCustomFoodModal, setShowCustomFoodModal] = useState(false);
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();

  // Format date for display
  const formatDisplayDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
    });
  };

  // Check if a date is today
  const isToday = (date: Date) => {
    const today = new Date();
    return date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear();
  };

  // Navigate to previous day
  const goToPreviousDay = () => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() - 1);
    setSelectedDate(newDate);
  };

  // Navigate to next day
  const goToNextDay = () => {
    const newDate = new Date(selectedDate);
    newDate.setDate(newDate.getDate() + 1);
    setSelectedDate(newDate);
  };

  // Go to today
  const goToToday = () => {
    setSelectedDate(new Date());
  };

  // Filter meals by selected date
  const filteredMeals = useMemo(() => {
    return userData?.calorieEntries?.filter((meal) => {
      if (!meal) return false;
      const mealDate = new Date(meal.date);
      return mealDate.getDate() === selectedDate.getDate() &&
        mealDate.getMonth() === selectedDate.getMonth() &&
        mealDate.getFullYear() === selectedDate.getFullYear();
    }) as MealEntry[] || [];
  }, [userData?.calorieEntries, selectedDate]);

  // Calculate daily totals for filtered meals
  const dailyTotals = useMemo(() => {
    return filteredMeals.reduce((acc, meal: Partial<MealEntry>) => ({
      calories: acc.calories + (meal.calories || 0),
      protein: acc.protein + (meal.macros?.protein || 0),
      carbs: acc.carbs + (meal.macros?.carbs || 0),
      fats: acc.fats + (meal.macros?.fats || 0),
    }), { calories: 0, protein: 0, carbs: 0, fats: 0 });
  }, [filteredMeals]);

  // Group meals by meal time
  const mealsByTime = useMemo(() => {
    return filteredMeals.reduce<Record<string, MealEntry[]>>((acc, meal: Partial<MealEntry>) => {
      if (!meal.macros || !meal.mealType) return acc;
      if (!acc[meal.mealType]) {
        acc[meal.mealType] = [];
      }
      acc[meal.mealType].push(meal as MealEntry);
      return acc;
    }, {});
  }, [filteredMeals]);

  // Calculate total macros for a meal time
  const calculateMealTimeTotals = (meals: MealEntry[]) => {
    return meals.reduce((acc, meal) => ({
      calories: acc.calories + meal.calories,
      protein: acc.protein + meal.macros.protein,
      carbs: acc.carbs + meal.macros.carbs,
      fats: acc.fats + meal.macros.fats,
    }), { calories: 0, protein: 0, carbs: 0, fats: 0 });
  };

  // Calculate macro percentage and get color
  const getMacroColor = (actual: number, target: number) => {
    const percentage = (actual / target) * 100;
    if (percentage < 80) return COLORS.error;
    if (percentage < 90) return COLORS.warning;
    if (percentage <= 110) return COLORS.success;
    return COLORS.error;
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };

  const formatTimeToAMPM = (timeStr: string) => {
    const [hours, minutes] = timeStr.split(':');
    const hour = parseInt(hours);
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const hour12 = hour % 12 || 12;
    return `${hour12}:${minutes} ${ampm}`;
  };

  const handleAddMeal = async () => {
    try {
      const protein = parseFloat(newMeal.protein);
      const carbs = parseFloat(newMeal.carbs);
      const fats = parseFloat(newMeal.fats);
      
      // Calculate calories from macros
      const calories = (protein * 4) + (carbs * 4) + (fats * 9);
      
      const newMealEntry: MealEntry = {
        id: Date.now().toString(),
        date: selectedDate.toISOString(),
        food: newMeal.food,
        calories,
        mealType: selectedMealTime,
        macros: {
          protein,
          carbs,
          fats,
        },
      };

      const updatedEntries = [...(userData.calorieEntries || []), newMealEntry];
      const updatedUserData = {
        ...userData,
        calorieEntries: updatedEntries,
      };

      await updateDoc(doc(db, 'users', userData.uid), {
        calorieEntries: updatedEntries,
      });

      onUpdateUserData(updatedUserData);
      setShowAddMealModal(false);
      setNewMeal({ food: '', protein: '', carbs: '', fats: '' });
    } catch (error) {
      console.error('Error adding meal:', error);
    }
  };

  const handleDeleteMeal = async (mealId: string) => {
    Alert.alert(
      "Delete Meal",
      "Are you sure you want to delete this meal?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            const updatedEntries = userData.calorieEntries?.filter(meal => meal.id !== mealId) || [];
            const updatedUserData = {
              ...userData,
              calorieEntries: updatedEntries,
            };
            
            await updateDoc(doc(db, 'users', userData.uid), {
              calorieEntries: updatedEntries,
            });
            
            onUpdateUserData(updatedUserData);
          }
        }
      ]
    );
  };

  const handleEditMeal = async () => {
    if (!selectedMeal) return;
    
    try {
      const protein = parseFloat(newMeal.protein);
      const carbs = parseFloat(newMeal.carbs);
      const fats = parseFloat(newMeal.fats);
      const calories = (protein * 4) + (carbs * 4) + (fats * 9);
      
      const updatedMeal: MealEntry = {
        ...selectedMeal,
        food: newMeal.food,
        calories,
        macros: { protein, carbs, fats },
      };

      const updatedEntries = userData.calorieEntries?.map(meal => 
        meal.id === selectedMeal.id ? updatedMeal : meal
      ) || [];

      const updatedUserData = {
        ...userData,
        calorieEntries: updatedEntries,
      };

      await updateDoc(doc(db, 'users', userData.uid), {
        calorieEntries: updatedEntries,
      });

      onUpdateUserData(updatedUserData);
      setShowEditMealModal(false);
      setSelectedMeal(null);
      setNewMeal({ food: '', protein: '', carbs: '', fats: '' });
    } catch (error) {
      console.error('Error editing meal:', error);
    }
  };

  const renderProgressBar = (current: number, target: number, label: string, color: string) => {
    const percentage = Math.min((current / target) * 100, 100);
    return (
      <View style={styles.progressContainer}>
        <View style={styles.progressLabelContainer}>
          <Text style={styles.progressLabel}>{label}</Text>
          <Text style={styles.progressValue}>
            {current.toFixed(1)} / {target.toFixed(1)} {label === 'Calories' ? 'kcal' : 'g'}
          </Text>
        </View>
        <View style={styles.progressBar}>
          <View style={[styles.progressFill, { width: `${percentage}%`, backgroundColor: color }]} />
        </View>
      </View>
    );
  };

  const renderMealItem = (meal: MealEntry) => (
    <View key={meal.id} style={styles.mealItem}>
      <View style={styles.mealItemHeader}>
        <Text style={styles.mealName}>{meal.food}</Text>
        <View style={styles.mealItemActions}>
          <TouchableOpacity 
            onPress={() => {
              setSelectedMeal(meal);
              setNewMeal({
                food: meal.food,
                protein: meal.macros.protein.toString(),
                carbs: meal.macros.carbs.toString(),
                fats: meal.macros.fats.toString(),
              });
              setShowEditMealModal(true);
            }}
            style={styles.actionButton}
          >
            <Ionicons name="pencil" size={18} color={COLORS.primary} />
          </TouchableOpacity>
          <TouchableOpacity 
            onPress={() => handleDeleteMeal(meal.id)}
            style={styles.actionButton}
          >
            <Ionicons name="trash" size={18} color={COLORS.error} />
          </TouchableOpacity>
        </View>
      </View>
      <Text style={styles.mealMacros}>
        P: {meal.macros.protein}g | C: {meal.macros.carbs}g | F: {meal.macros.fats}g
      </Text>
      <Text style={styles.mealTimestamp}>
        Added: {formatDate(meal.date)} at {formatTime(meal.date)}
      </Text>
    </View>
  );

  const renderMealTimeCard = (mealTime: string, targetMeal: typeof userData.settings.mealPlan.meals[0]) => {
    const meals = mealsByTime[mealTime] || [];
    const totals = calculateMealTimeTotals(meals);

    return (
      <View style={styles.mealTimeCard}>
        <View style={styles.mealTimeHeader}>
          <Text style={styles.mealTimeTitle}>{mealTime}</Text>
          <Text style={styles.mealTimeSchedule}>{formatTimeToAMPM(targetMeal.time)}</Text>
          <TouchableOpacity 
            style={styles.addMealButton}
            onPress={() => {
              navigation.navigate('FoodSearch', {
                onSelectFood: (food: USDAFoodItem) => {
                  setSelectedMeal(null);
                  setNewMeal({
                    food: food.description,
                    protein: food.nutrients.protein.toString(),
                    carbs: food.nutrients.carbohydrates.toString(),
                    fats: food.nutrients.fat.toString(),
                  });
                  setSelectedMealTime(mealTime);
                  setShowAddMealModal(true);
                },
              });
            }}
          >
            <Text style={styles.addMealButtonText}>+ Add Food</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.macroGoals}>
          <Text style={styles.macroGoalTitle}>Target:</Text>
          <Text style={styles.macroGoalText}>
            P: {targetMeal.macros.protein}g | C: {targetMeal.macros.carbs}g | F: {targetMeal.macros.fats}g
          </Text>
        </View>

        {meals.map((meal, i) => (
          <React.Fragment key={`meal-item-${meal.id}-${i}`}>
            {renderMealItem(meal)}
          </React.Fragment>
        ))}

        <View style={styles.totalMacros}>
          <Text style={styles.totalTitle}>Total:</Text>
          <Text style={[
            styles.macroText,
            { color: getMacroColor(totals.protein, targetMeal.macros.protein) }
          ]}>P: {totals.protein.toFixed(1)}g</Text>
          <Text style={[
            styles.macroText,
            { color: getMacroColor(totals.carbs, targetMeal.macros.carbs) }
          ]}>C: {totals.carbs.toFixed(1)}g</Text>
          <Text style={[
            styles.macroText,
            { color: getMacroColor(totals.fats, targetMeal.macros.fats) }
          ]}>F: {totals.fats.toFixed(1)}g</Text>
        </View>
      </View>
    );
  };

  const handleSearch = async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }
    
    setIsSearching(true);
    setSearchResults([]); // Clear previous results while searching
    
    try {
      console.log('Searching for:', query);
      const results = await searchFoods(query);
      console.log('Search results:', results);
      setSearchResults(results);
    } catch (error) {
      console.error('Error searching foods:', error);
      Alert.alert(
        'Error',
        'Failed to search foods. Please try again.',
        [{ text: 'OK' }]
      );
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleFoodSelect = (food: USDAFoodItem) => {
    // Add to recent foods if not already present
    setRecentFoods(prev => {
      const isExisting = prev.some(f => f.fdcId === food.fdcId);
      if (!isExisting) {
        const newRecent = [food, ...prev].slice(0, 5); // Keep only last 5
        AsyncStorage.setItem('recentFoods', JSON.stringify(newRecent))
          .catch(error => console.error('Error saving recent foods:', error));
        return newRecent;
      }
      return prev;
    });

    // Pre-fill the add meal form with the selected food's data
    setSelectedMeal(null);
    setNewMeal({
      food: food.description,
      protein: food.nutrients.protein.toString(),
      carbs: food.nutrients.carbohydrates.toString(),
      fats: food.nutrients.fat.toString(),
    });
    
    setShowFoodSearchModal(false);
    setShowAddMealModal(true);
  };

  const toggleFavorite = (food: USDAFoodItem) => {
    setFavoriteFoods(prev => {
      const isFavorite = prev.some(f => f.fdcId === food.fdcId);
      let newFavorites;
      
      if (isFavorite) {
        newFavorites = prev.filter(f => f.fdcId !== food.fdcId);
      } else {
        newFavorites = [...prev, food];
      }

      AsyncStorage.setItem('favoriteFoods', JSON.stringify(newFavorites))
        .catch(error => console.error('Error saving favorite foods:', error));
      
      return newFavorites;
    });
  };

  // Load saved foods on mount
  useEffect(() => {
    const loadSavedFoods = async () => {
      try {
        const [recentJson, favoritesJson] = await Promise.all([
          AsyncStorage.getItem('recentFoods'),
          AsyncStorage.getItem('favoriteFoods')
        ]);

        if (recentJson) setRecentFoods(JSON.parse(recentJson));
        if (favoritesJson) setFavoriteFoods(JSON.parse(favoritesJson));
      } catch (error) {
        console.error('Error loading saved foods:', error);
      }
    };

    loadSavedFoods();
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <LogoImage size={120} withText={true} style={styles.logo} />
        <Text style={styles.headerTitle}>Nutrition Tracker</Text>
        <DateSelector 
          selectedDate={selectedDate} 
          onDateChange={setSelectedDate}
        />
      </View>

      <ScrollView style={styles.content}>
        <View style={styles.statsCard}>
          <Text style={styles.statsTitle}>Daily Progress</Text>
          {renderProgressBar(
            dailyTotals.calories,
            userData.settings.calorieGoal,
            'Calories',
            COLORS.primary
          )}
          {renderProgressBar(
            dailyTotals.protein,
            (userData.settings.calorieGoal * userData.settings.macroSplit.protein / 100) / 4,
            'Protein',
            COLORS.success
          )}
          {renderProgressBar(
            dailyTotals.carbs,
            (userData.settings.calorieGoal * userData.settings.macroSplit.carbs / 100) / 4,
            'Carbs',
            COLORS.warning
          )}
          {renderProgressBar(
            dailyTotals.fats,
            (userData.settings.calorieGoal * userData.settings.macroSplit.fats / 100) / 9,
            'Fats',
            COLORS.error
          )}
        </View>

        {userData.settings.mealPlan.meals.map((meal, index) => 
          <React.Fragment key={`meal-time-${index}`}>
            {renderMealTimeCard(`Meal ${index + 1}`, meal)}
          </React.Fragment>
        )}
      </ScrollView>

      <Modal
        visible={showAddMealModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowAddMealModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>
              {selectedMeal ? 'Edit Meal' : 'Add Meal'}
            </Text>

            <View style={styles.inputGroup}>
              <Text style={styles.label}>Food</Text>
              <TextInput
                style={styles.input}
                value={newMeal.food}
                onChangeText={(text) => setNewMeal({ ...newMeal, food: text })}
                placeholder="Enter food name"
                placeholderTextColor={COLORS.subtext}
              />
            </View>

            <View style={styles.macroInputs}>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Protein (g)</Text>
                <TextInput
                  style={styles.input}
                  value={newMeal.protein}
                  onChangeText={(text) => setNewMeal({ ...newMeal, protein: text })}
                  keyboardType="numeric"
                  placeholder="0"
                  placeholderTextColor={COLORS.subtext}
                />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>Carbs (g)</Text>
                <TextInput
                  style={styles.input}
                  value={newMeal.carbs}
                  onChangeText={(text) => setNewMeal({ ...newMeal, carbs: text })}
                  keyboardType="numeric"
                  placeholder="0"
                  placeholderTextColor={COLORS.subtext}
                />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.label}>Fats (g)</Text>
                <TextInput
                  style={styles.input}
                  value={newMeal.fats}
                  onChangeText={(text) => setNewMeal({ ...newMeal, fats: text })}
                  keyboardType="numeric"
                  placeholder="0"
                  placeholderTextColor={COLORS.subtext}
                />
              </View>
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => {
                  setShowAddMealModal(false);
                  setSelectedMeal(null);
                  setNewMeal({
                    food: '',
                    protein: '',
                    carbs: '',
                    fats: '',
                  });
                }}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.modalButton, styles.saveButton]}
                onPress={handleAddMeal}
              >
                <Text style={styles.saveButtonText}>
                  {selectedMeal ? 'Save Changes' : 'Add Meal'}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <Modal
        visible={showEditMealModal}
        animationType="slide"
        transparent={true}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Edit Food</Text>
            
            <TextInput
              style={styles.input}
              placeholder="Food name"
              value={newMeal.food}
              onChangeText={(text) => setNewMeal(prev => ({ ...prev, food: text }))}
            />
            
            <View style={styles.macroInputs}>
              <View style={styles.macroInputContainer}>
                <Text style={styles.macroInputLabel}>Protein (g)</Text>
                <TextInput
                  style={styles.macroInput}
                  keyboardType="numeric"
                  value={newMeal.protein}
                  onChangeText={(text) => setNewMeal(prev => ({ ...prev, protein: text }))}
                />
              </View>
              
              <View style={styles.macroInputContainer}>
                <Text style={styles.macroInputLabel}>Carbs (g)</Text>
                <TextInput
                  style={styles.macroInput}
                  keyboardType="numeric"
                  value={newMeal.carbs}
                  onChangeText={(text) => setNewMeal(prev => ({ ...prev, carbs: text }))}
                />
              </View>
              
              <View style={styles.macroInputContainer}>
                <Text style={styles.macroInputLabel}>Fats (g)</Text>
                <TextInput
                  style={styles.macroInput}
                  keyboardType="numeric"
                  value={newMeal.fats}
                  onChangeText={(text) => setNewMeal(prev => ({ ...prev, fats: text }))}
                />
              </View>
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity 
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => {
                  setShowEditMealModal(false);
                  setSelectedMeal(null);
                  setNewMeal({ food: '', protein: '', carbs: '', fats: '' });
                }}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.modalButton, styles.addButton]}
                onPress={handleEditMeal}
              >
                <Text style={styles.addButtonText}>Save Changes</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      <Modal
        visible={showFoodSearchModal}
        animationType="slide"
        transparent={true}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Search Foods</Text>
            
            <View style={styles.searchContainer}>
              <TextInput
                style={styles.searchInput}
                placeholder="Search for a food..."
                value={searchQuery}
                onChangeText={setSearchQuery}
                onSubmitEditing={() => handleSearch(searchQuery)}
                returnKeyType="search"
                autoFocus
              />
              <TouchableOpacity 
                style={[styles.searchButton, searchQuery.length < 2 && styles.searchButtonDisabled]}
                onPress={() => handleSearch(searchQuery)}
                disabled={searchQuery.length < 2}
              >
                <Text style={[styles.searchButtonText, searchQuery.length < 2 && styles.searchButtonTextDisabled]}>
                  Search
                </Text>
              </TouchableOpacity>
              <TouchableOpacity 
                style={styles.closeButton}
                onPress={() => {
                  setShowFoodSearchModal(false);
                  setSearchQuery('');
                  setSearchResults([]);
                }}
              >
                <Ionicons name="close" size={24} color={COLORS.text} />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.searchResults}>
              {isSearching ? (
                <ActivityIndicator size="large" color={COLORS.primary} />
              ) : (
                <>
                  {searchQuery.length < 2 ? (
                    <>
                      <View style={styles.section}>
                        <Text style={styles.sectionTitle}>Recent Foods</Text>
                        {recentFoods.length > 0 ? (
                          recentFoods.map(food => (
                            <FoodItem
                              key={food.fdcId}
                              food={food}
                              onSelect={handleFoodSelect}
                              onFavorite={toggleFavorite}
                              isFavorite={favoriteFoods.some(f => f.fdcId === food.fdcId)}
                            />
                          ))
                        ) : (
                          <Text style={styles.noResultsText}>No recent foods</Text>
                        )}
                      </View>
                      
                      <View style={styles.section}>
                        <Text style={styles.sectionTitle}>Favorite Foods</Text>
                        {favoriteFoods.length > 0 ? (
                          favoriteFoods.map(food => (
                            <FoodItem
                              key={food.fdcId}
                              food={food}
                              onSelect={handleFoodSelect}
                              onFavorite={toggleFavorite}
                              isFavorite={true}
                            />
                          ))
                        ) : (
                          <Text style={styles.noResultsText}>No favorite foods</Text>
                        )}
                      </View>
                    </>
                  ) : searchResults.length > 0 ? (
                    <View style={styles.searchResultsContainer}>
                      <Text style={styles.sectionTitle}>Search Results</Text>
                      {searchResults.map(food => (
                        <FoodItem
                          key={food.fdcId}
                          food={food}
                          onSelect={handleFoodSelect}
                          onFavorite={toggleFavorite}
                          isFavorite={favoriteFoods.some(f => f.fdcId === food.fdcId)}
                        />
                      ))}
                    </View>
                  ) : (
                    <View style={styles.noResults}>
                      <Text style={styles.noResultsText}>No foods found</Text>
                    </View>
                  )}
                </>
              )}
            </ScrollView>

            <TouchableOpacity
              style={styles.addCustomButton}
              onPress={() => {
                setShowFoodSearchModal(false);
                setShowCustomFoodModal(true);
              }}
            >
              <Text style={styles.addCustomButtonText}>Create Custom Food</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Modal
        visible={showCustomFoodModal}
        animationType="slide"
        transparent={true}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Create Custom Food</Text>
            
            <TextInput
              style={styles.input}
              placeholder="Food name"
              value={newMeal.food}
              onChangeText={(text) => setNewMeal(prev => ({ ...prev, food: text }))}
            />
            
            <View style={styles.macroInputs}>
              <View style={styles.macroInputContainer}>
                <Text style={styles.macroInputLabel}>Protein (g)</Text>
                <TextInput
                  style={styles.macroInput}
                  keyboardType="numeric"
                  value={newMeal.protein}
                  onChangeText={(text) => setNewMeal(prev => ({ ...prev, protein: text }))}
                />
              </View>
              
              <View style={styles.macroInputContainer}>
                <Text style={styles.macroInputLabel}>Carbs (g)</Text>
                <TextInput
                  style={styles.macroInput}
                  keyboardType="numeric"
                  value={newMeal.carbs}
                  onChangeText={(text) => setNewMeal(prev => ({ ...prev, carbs: text }))}
                />
              </View>
              
              <View style={styles.macroInputContainer}>
                <Text style={styles.macroInputLabel}>Fats (g)</Text>
                <TextInput
                  style={styles.macroInput}
                  keyboardType="numeric"
                  value={newMeal.fats}
                  onChangeText={(text) => setNewMeal(prev => ({ ...prev, fats: text }))}
                />
              </View>
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity 
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => {
                  setShowCustomFoodModal(false);
                  setNewMeal({ food: '', protein: '', carbs: '', fats: '' });
                }}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.modalButton, styles.addButton]}
                onPress={() => {
                  const customFood: USDAFoodItem = {
                    fdcId: `custom_${Date.now()}`,
                    description: newMeal.food,
                    nutrients: {
                      protein: parseFloat(newMeal.protein) || 0,
                      carbohydrates: parseFloat(newMeal.carbs) || 0,
                      fat: parseFloat(newMeal.fats) || 0,
                      calories: (parseFloat(newMeal.protein) * 4) + 
                              (parseFloat(newMeal.carbs) * 4) + 
                              (parseFloat(newMeal.fats) * 9) || 0,
                    }
                  };
                  setCustomFoods(prev => [...prev, customFood]);
                  AsyncStorage.setItem('customFoods', JSON.stringify(customFoods))
                    .catch(error => console.error('Error saving custom foods:', error));
                  handleFoodSelect(customFood);
                  setShowCustomFoodModal(false);
                }}
              >
                <Text style={styles.addButtonText}>Create & Add</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  header: {
    padding: 20,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: SIZES.xLarge,
    fontWeight: 'bold',
    color: COLORS.card,
    marginBottom: 10,
    textAlign: 'center',
  },
  content: {
    flex: 1,
  },
  statsCard: {
    backgroundColor: COLORS.card,
    margin: 15,
    borderRadius: 10,
    padding: 15,
    ...SHADOWS.small,
  },
  statsTitle: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 15,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  statLabel: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
  },
  mealTimeCard: {
    backgroundColor: COLORS.card,
    margin: 15,
    marginTop: 0,
    borderRadius: 10,
    padding: 15,
    ...SHADOWS.small,
  },
  mealTimeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  mealTimeTitle: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  mealTimeSchedule: {
    fontSize: SIZES.small,
    color: COLORS.primary,
    fontWeight: 'bold',
  },
  addMealButton: {
    backgroundColor: COLORS.secondary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  addMealButtonText: {
    color: COLORS.card,
    fontSize: SIZES.small,
    fontWeight: 'bold',
  },
  macroGoals: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  macroGoalTitle: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginRight: 8,
  },
  macroGoalText: {
    fontSize: SIZES.small,
    color: COLORS.text,
  },
  mealItem: {
    backgroundColor: COLORS.background,
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
  },
  mealName: {
    fontSize: SIZES.medium,
    color: COLORS.text,
    marginBottom: 4,
  },
  mealMacros: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
  },
  totalMacros: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
  },
  totalTitle: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginRight: 8,
  },
  macroText: {
    fontSize: SIZES.small,
    fontWeight: 'bold',
    marginRight: 12,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: COLORS.card,
    borderRadius: 15,
    padding: 20,
    ...SHADOWS.medium,
  },
  modalTitle: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    backgroundColor: COLORS.background,
    borderRadius: 8,
    padding: 12,
    marginBottom: 15,
    fontSize: SIZES.medium,
  },
  macroInputs: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  macroInputContainer: {
    flex: 1,
    marginHorizontal: 5,
  },
  macroInputLabel: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginBottom: 5,
  },
  macroInput: {
    backgroundColor: COLORS.background,
    borderRadius: 8,
    padding: 12,
    fontSize: SIZES.medium,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  cancelButton: {
    backgroundColor: COLORS.error,
  },
  addButton: {
    backgroundColor: COLORS.primary,
  },
  cancelButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  addButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  progressContainer: {
    marginBottom: 15,
  },
  progressLabelContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  progressLabel: {
    fontSize: SIZES.small,
    color: COLORS.text,
    fontWeight: 'bold',
  },
  progressValue: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
  },
  progressBar: {
    height: 8,
    backgroundColor: COLORS.border,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  mealItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  mealItemActions: {
    flexDirection: 'row',
    gap: 8,
  },
  actionButton: {
    padding: 4,
  },
  mealTimestamp: {
    fontSize: SIZES.xSmall,
    color: COLORS.subtext,
    fontStyle: 'italic',
    marginTop: 4,
  },
  dateNavigation: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
  },
  dateButton: {
    padding: 8,
  },
  dateContainer: {
    alignItems: 'center',
    flex: 1,
  },
  dateText: {
    fontSize: SIZES.medium,
    color: COLORS.card,
    fontWeight: 'bold',
  },
  todayHint: {
    fontSize: SIZES.xSmall,
    color: COLORS.card,
    opacity: 0.8,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 8,
  },
  searchInput: {
    flex: 1,
    padding: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 8,
    backgroundColor: COLORS.background,
  },
  searchButton: {
    backgroundColor: COLORS.primary,
    padding: 10,
    borderRadius: 8,
    minWidth: 70,
    alignItems: 'center',
  },
  searchButtonDisabled: {
    backgroundColor: COLORS.border,
  },
  searchButtonText: {
    color: COLORS.card,
    fontWeight: 'bold',
  },
  searchButtonTextDisabled: {
    color: COLORS.subtext,
  },
  closeButton: {
    padding: 10,
  },
  searchResults: {
    flex: 1,
    maxHeight: '70%',
  },
  section: {
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 5,
  },
  searchResultsContainer: {
    flex: 1,
  },
  addCustomButton: {
    backgroundColor: COLORS.primary,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  addCustomButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
  logo: {
    marginBottom: 10,
  },
  noResults: {
    padding: 20,
    alignItems: 'center',
  },
  noResultsText: {
    fontSize: SIZES.medium,
    color: COLORS.subtext,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: SIZES.small,
    color: COLORS.text,
    marginBottom: 8,
  },
  saveButton: {
    backgroundColor: COLORS.success,
  },
  saveButtonText: {
    color: COLORS.card,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
  },
});

export default NutritionScreen; 