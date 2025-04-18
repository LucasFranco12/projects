import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';

interface MealEntry {
  id: string;
  time: string;
  food: string;
  protein: string;
  carbs: string;
  fats: string;
}

interface MealCardProps {
  meal: MealEntry;
  onEdit: () => void;
  onDelete: () => void;
}

const MealCard = ({ meal, onEdit, onDelete }: MealCardProps) => {
  const calculateCalories = () => {
    const protein = parseFloat(meal.protein) || 0;
    const carbs = parseFloat(meal.carbs) || 0;
    const fats = parseFloat(meal.fats) || 0;
    return Math.round((protein * 4) + (carbs * 4) + (fats * 9));
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text style={styles.foodName}>{meal.food}</Text>
          <Text style={styles.time}>{meal.time}</Text>
        </View>
        <View style={styles.actions}>
          <TouchableOpacity onPress={onEdit} style={styles.actionButton}>
            <Ionicons name="pencil" size={20} color={COLORS.primary} />
          </TouchableOpacity>
          <TouchableOpacity onPress={onDelete} style={styles.actionButton}>
            <Ionicons name="trash" size={20} color={COLORS.error} />
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.macros}>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{meal.protein}g</Text>
          <Text style={styles.macroLabel}>Protein</Text>
        </View>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{meal.carbs}g</Text>
          <Text style={styles.macroLabel}>Carbs</Text>
        </View>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{meal.fats}g</Text>
          <Text style={styles.macroLabel}>Fats</Text>
        </View>
        <View style={styles.macroItem}>
          <Text style={styles.macroValue}>{calculateCalories()}</Text>
          <Text style={styles.macroLabel}>Calories</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: COLORS.card,
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    ...SHADOWS.small,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  foodName: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 4,
  },
  time: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    padding: 4,
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

export default MealCard; 