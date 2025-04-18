import React from 'react';
import { StyleSheet, Text, View, FlatList, TouchableOpacity, ScrollView } from 'react-native';
import { COLORS, SIZES, SHADOWS } from '../constants/theme';
import { UserData } from '../App';
import LogoImage from '../components/LogoImage';

interface Exercise {
  name: string;
  sets: number;
  reps: number;
  weight?: number;
}

interface Workout {
  id: string;
  date: string;
  exercises: Exercise[];
}

interface WorkoutsScreenProps {
  userData: UserData;
}

const WorkoutsScreen = ({ userData }: WorkoutsScreenProps) => {
  // Sample workout data
  const workouts = userData?.workouts || [
    {
      id: '1',
      date: '2023-05-15',
      exercises: [
        { name: 'Bench Press', sets: 3, reps: 10, weight: 135 },
        { name: 'Squats', sets: 3, reps: 12, weight: 185 },
      ]
    }
  ];
  
  const renderWorkoutItem = ({ item }: { item: Workout }) => (
    <View style={styles.workoutCard}>
      <Text style={styles.dateText}>{new Date(item.date).toLocaleDateString()}</Text>
      
      {item.exercises.map((exercise: Exercise, index: number) => (
        <View key={index} style={styles.exerciseItem}>
          <Text style={styles.exerciseName}>{exercise.name}</Text>
          <Text style={styles.exerciseDetails}>
            {exercise.sets} sets × {exercise.reps} reps
            {exercise.weight ? ` × ${exercise.weight}lbs` : ''}
          </Text>
        </View>
      ))}
    </View>
  );
  
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <LogoImage size={120} withText={true} style={styles.logo} />
        <Text style={styles.headerTitle}>Workouts</Text>
      </View>
      
      <ScrollView style={styles.content}>
        <TouchableOpacity style={styles.addButton}>
          <Text style={styles.addButtonText}>+ Add New Workout</Text>
        </TouchableOpacity>
        
        {workouts.length > 0 ? (
          <FlatList
            data={workouts}
            renderItem={renderWorkoutItem}
            keyExtractor={(item: Workout) => item.id}
            contentContainerStyle={styles.listContainer}
          />
        ) : (
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>No workouts logged yet.</Text>
            <Text style={styles.emptySubtext}>Start tracking your fitness journey today!</Text>
          </View>
        )}
      </ScrollView>
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
  logo: {
    marginBottom: 10,
  },
  headerTitle: {
    fontSize: SIZES.large,
    color: COLORS.card,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
    padding: 15,
  },
  addButton: {
    backgroundColor: COLORS.secondary,
    margin: 15,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    ...SHADOWS.small,
  },
  addButtonText: {
    color: COLORS.card,
    fontWeight: 'bold',
    fontSize: SIZES.medium,
  },
  listContainer: {
    padding: 15,
  },
  workoutCard: {
    backgroundColor: COLORS.card,
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    ...SHADOWS.small,
  },
  dateText: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginBottom: 10,
  },
  exerciseItem: {
    borderLeftWidth: 2,
    borderLeftColor: COLORS.accent,
    paddingLeft: 10,
    marginBottom: 8,
  },
  exerciseName: {
    fontSize: SIZES.medium,
    fontWeight: 'bold',
    color: COLORS.text,
  },
  exerciseDetails: {
    fontSize: SIZES.small,
    color: COLORS.subtext,
    marginTop: 3,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: SIZES.large,
    fontWeight: 'bold',
    color: COLORS.text,
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: SIZES.medium,
    color: COLORS.subtext,
    textAlign: 'center',
  },
});

export default WorkoutsScreen; 