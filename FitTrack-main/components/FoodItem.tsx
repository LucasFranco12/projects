import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import { USDAFoodItem } from '../utils/foodDatabase';

interface FoodItemProps {
  food: USDAFoodItem;
  onSelect: (food: USDAFoodItem) => void;
  onFavorite: (food: USDAFoodItem) => void;
  isFavorite: boolean;
}

const FoodItem = ({ food, onSelect, onFavorite, isFavorite }: FoodItemProps) => {
  const formatNutrient = (value: number) => {
    return value.toFixed(1);
  };

  return (
    <TouchableOpacity style={styles.container} onPress={() => onSelect(food)}>
      <View style={styles.header}>
        <Text style={styles.foodName}>{food.description}</Text>
        <TouchableOpacity onPress={() => onFavorite(food)} style={styles.favoriteButton}>
          <Ionicons
            name={isFavorite ? "star" : "star-outline"}
            size={24}
            color={isFavorite ? COLORS.primary : COLORS.text}
          />
        </TouchableOpacity>
      </View>

      {food.servingSize && (
        <Text style={styles.servingSize}>
          Serving: {food.servingSize} {food.servingSizeUnit}
        </Text>
      )}

      <View style={styles.macros}>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{formatNutrient(food.nutrients.protein)}g</Text>
          <Text style={styles.macroLabel}>Protein</Text>
        </View>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{formatNutrient(food.nutrients.carbohydrates)}g</Text>
          <Text style={styles.macroLabel}>Carbs</Text>
        </View>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{formatNutrient(food.nutrients.fat)}g</Text>
          <Text style={styles.macroLabel}>Fat</Text>
        </View>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{Math.round(food.nutrients.calories)}</Text>
          <Text style={styles.macroLabel}>Calories</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.card,
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    ...SHADOWS.small,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  foodName: {
    flex: 1,
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginRight: 8,
  },
  favoriteButton: {
    padding: 4,
  },
  servingSize: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginBottom: 8,
  },
  macros: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  macroItem: {
    alignItems: 'center',
  },
  macroValue: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  macroLabel: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginTop: 4,
  },
});

export default FoodItem; 