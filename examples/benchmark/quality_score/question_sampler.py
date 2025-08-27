import sqlite3
import random
from typing import List, Dict, Literal
from pathlib import Path

def sample_questions(
    degree_of_repetition: Literal['HIGH', 'LOW', 'MIXED'],
    number_of_questions: int,
    db_path: str = "paraphrased_questions.db"
) -> List[Dict[str, any]]:
    """
    Sample questions from the paraphrased questions database based on degree of repetition.
    
    Args:
        degree_of_repetition: 'HIGH' (22-32 questions per cluster), 'LOW' (1-12 questions per cluster), 
                             or 'MIXED' (varied distribution)
        number_of_questions: Total number of questions to return (1-3200)
        db_path: Path to the SQLite database file
        
    Returns:
        List of dictionaries containing question_id, cluster_id, and question text
        
    Raises:
        ValueError: If number_of_questions is out of range or parameters are invalid
        FileNotFoundError: If database file doesn't exist
    """
    
    # Validate inputs
    if not 1 <= number_of_questions <= 3200:
        raise ValueError("number_of_questions must be between 1 and 3200")
    
    if degree_of_repetition not in ['HIGH', 'LOW', 'MIXED']:
        raise ValueError("degree_of_repetition must be 'HIGH', 'LOW', or 'MIXED'")
    
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    cursor = conn.cursor()
    
    try:
        # Get all available cluster IDs (clusters range from 5 to 104 based on our check)
        cursor.execute("SELECT DISTINCT cluster_id FROM paraphrased_questions ORDER BY cluster_id")
        all_clusters = [row[0] for row in cursor.fetchall()]
        
        # Randomly shuffle clusters to ensure random selection
        random.shuffle(all_clusters)
        
        selected_questions = []
        questions_collected = 0
        clusters_used = []
        
        # Define questions per cluster based on degree of repetition
        if degree_of_repetition == 'HIGH':
            questions_per_cluster_range = (22, 32)
        elif degree_of_repetition == 'LOW':
            questions_per_cluster_range = (1, 12)
        else:  # MIXED
            questions_per_cluster_range = (1, 32)
        
        # Sample questions from clusters
        for cluster_id in all_clusters:
            if questions_collected >= number_of_questions:
                break
                
            # Determine how many questions to take from this cluster
            min_questions, max_questions = questions_per_cluster_range
            
            # For MIXED mode, vary the distribution more
            if degree_of_repetition == 'MIXED':
                # Create a mixed distribution: some clusters with few questions, some with many
                if random.random() < 0.3:  # 30% chance for high repetition
                    questions_from_cluster = random.randint(22, 32)
                elif random.random() < 0.5:  # 35% chance for medium repetition (remaining 70% * 0.5)
                    questions_from_cluster = random.randint(8, 21)
                else:  # 35% chance for low repetition
                    questions_from_cluster = random.randint(1, 7)
            else:
                questions_from_cluster = random.randint(min_questions, max_questions)
            
            # Calculate remaining questions needed
            remaining_questions = number_of_questions - questions_collected
            
            # For HIGH and LOW modes, try to respect minimum bounds when possible
            if degree_of_repetition in ['HIGH', 'LOW'] and remaining_questions < min_questions:
                # If we can't meet the minimum, take what we can but prefer to skip this cluster
                # and let the remaining questions be distributed among fewer clusters
                if remaining_questions >= min_questions // 2:  # Take at least half the minimum
                    questions_from_cluster = remaining_questions
                else:
                    continue  # Skip this cluster to maintain better distribution
            else:
                # Don't exceed the remaining questions needed
                questions_from_cluster = min(questions_from_cluster, remaining_questions)
            
            # Get random questions from this cluster
            cursor.execute("""
                SELECT question_id, cluster_id, question 
                FROM paraphrased_questions 
                WHERE cluster_id = ? 
                ORDER BY RANDOM() 
                LIMIT ?
            """, (cluster_id, questions_from_cluster))
            
            cluster_questions = cursor.fetchall()
            
            # Add questions to our selection
            for question_row in cluster_questions:
                selected_questions.append({
                    'question_id': question_row['question_id'],
                    'cluster_id': question_row['cluster_id'],
                    'question': question_row['question']
                })
                questions_collected += 1
            
            clusters_used.append(cluster_id)
            
            if questions_collected >= number_of_questions:
                break
        
        # Shuffle the final results to mix questions from different clusters
        random.shuffle(selected_questions)
        
        # Ensure we return exactly the requested number of questions
        selected_questions = selected_questions[:number_of_questions]
        
        return selected_questions
        
    finally:
        conn.close()


def get_sampling_stats(questions: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Get statistics about the sampled questions.
    
    Args:
        questions: List of question dictionaries returned by sample_questions
        
    Returns:
        Dictionary with statistics about clusters and distribution
    """
    if not questions:
        return {"total_questions": 0, "clusters_used": 0, "questions_per_cluster": {}}
    
    cluster_counts = {}
    for question in questions:
        cluster_id = question['cluster_id']
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    return {
        "total_questions": len(questions),
        "clusters_used": len(cluster_counts),
        "questions_per_cluster": cluster_counts,
        "avg_questions_per_cluster": sum(cluster_counts.values()) / len(cluster_counts),
        "min_questions_per_cluster": min(cluster_counts.values()),
        "max_questions_per_cluster": max(cluster_counts.values())
    }


