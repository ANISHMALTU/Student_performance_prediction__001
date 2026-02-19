from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import csv
import io
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

# Database Configuration
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
application.config['SECRET_KEY'] = 'your-secret-key-123'

db = SQLAlchemy(application)

# Database Model for storing predictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.String(20))
    race_ethnicity = db.Column(db.String(50))
    parental_level_of_education = db.Column(db.String(100))
    lunch = db.Column(db.String(50))
    test_preparation_course = db.Column(db.String(50))
    reading_score = db.Column(db.Float)
    writting_score = db.Column(db.Float)
    predicted_math_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'gender': self.gender,
            'race_ethnicity': self.race_ethnicity,
            'parental_level_of_education': self.parental_level_of_education,
            'lunch': self.lunch,
            'test_preparation_course': self.test_preparation_course,
            'reading_score': self.reading_score,
            'writting_score': self.writting_score,
            'predicted_math_score': self.predicted_math_score,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# Create database tables
with application.app_context():
    db.create_all()

app = application

# Route for home page
@app.route('/')
def index():
    # Get statistics for dashboard
    total_predictions = Prediction.query.count()
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
    
    # Calculate average predicted score
    avg_score = db.session.query(db.func.avg(Prediction.predicted_math_score)).scalar()
    avg_score = round(avg_score, 2) if avg_score else 0
    
    # Get predictions by gender
    gender_stats = db.session.query(
        Prediction.gender,
        db.func.count(Prediction.id),
        db.func.avg(Prediction.predicted_math_score)
    ).group_by(Prediction.gender).all()
    
    return render_template('index.html', 
                           total_predictions=total_predictions,
                           recent_predictions=recent_predictions,
                           avg_score=avg_score,
                           gender_stats=gender_stats)

# Route for prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get form data
        gender = request.form.get('gender')
        ethnicity = request.form.get('ethnicity')
        parental_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_prep = request.form.get('test_preparation_course')
        reading_score = float(request.form.get('reading_score', 0))
        writting_score = float(request.form.get('writting_score', 0))
        
        # Create custom data object
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_prep,
            reading_score=reading_score,
            writting_score=writting_score
        )
        
        pred_df = data.get_data_as_data_frame()
        
        # Make prediction
        predictionPipeline = PredictPipeline()
        result = predictionPipeline.predict(pred_df)[0]
        
        # Save prediction to database
        prediction = Prediction(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_prep,
            reading_score=reading_score,
            writting_score=writting_score,
            predicted_math_score=result
        )
        db.session.add(prediction)
        db.session.commit()
        
        flash(f'Prediction saved successfully!', 'success')
        
        return render_template('home.html', results=result)

# Route for dashboard
@app.route('/dashboard')
def dashboard():
    # Get all predictions with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Get statistics
    total_predictions = Prediction.query.count()
    avg_score = db.session.query(db.func.avg(Prediction.predicted_math_score)).scalar()
    avg_score = round(avg_score, 2) if avg_score else 0
    
    max_score = db.session.query(db.func.max(Prediction.predicted_math_score)).scalar()
    min_score = db.session.query(db.func.min(Prediction.predicted_math_score)).scalar()
    
    # Get gender distribution - convert Row objects to lists for JSON serialization
    gender_dist_raw = db.session.query(
        Prediction.gender,
        db.func.count(Prediction.id)
    ).group_by(Prediction.gender).all()
    gender_dist = [[row[0], row[1]] for row in gender_dist_raw]
    
    # Get test preparation course distribution - convert Row objects to lists
    test_prep_dist_raw = db.session.query(
        Prediction.test_preparation_course,
        db.func.count(Prediction.id)
    ).group_by(Prediction.test_preparation_course).all()
    test_prep_dist = [[row[0], row[1]] for row in test_prep_dist_raw]
    
    return render_template('dashboard.html',
                           predictions=predictions,
                           total_predictions=total_predictions,
                           avg_score=avg_score,
                           max_score=max_score,
                           min_score=min_score,
                           gender_dist=gender_dist,
                           test_prep_dist=test_prep_dist)

# API endpoint for getting predictions
@app.route('/api/predictions')
def api_predictions():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
    return jsonify([p.to_dict() for p in predictions])

# API endpoint for getting statistics
@app.route('/api/statistics')
def api_statistics():
    total = Prediction.query.count()
    avg = db.session.query(db.func.avg(Prediction.predicted_math_score)).scalar()
    max_score = db.session.query(db.func.max(Prediction.predicted_math_score)).scalar()
    min_score = db.session.query(db.func.min(Prediction.predicted_math_score)).scalar()
    
    return jsonify({
        'total_predictions': total,
        'average_score': round(avg, 2) if avg else 0,
        'max_score': max_score,
        'min_score': min_score
    })

# Route for deleting a prediction
@app.route('/delete/<int:id>')
def delete_prediction(id):
    prediction = Prediction.query.get_or_404(id)
    db.session.delete(prediction)
    db.session.commit()
    flash('Prediction deleted successfully!', 'success')
    return redirect(url_for('dashboard'))

# Route for clearing all predictions
@app.route('/clearall')
def clear_all():
    db.session.query(Prediction).delete()
    db.session.commit()
    flash('All predictions cleared!', 'success')
    return redirect(url_for('dashboard'))

# ============================================
# NEW ENDPOINTS FOR COLLEGE-READY FEATURES
# ============================================

# Search predictions endpoint
@app.route('/search')
def search_predictions():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Build search query
    search_query = Prediction.query
    if query:
        search_query = search_query.filter(
            (Prediction.gender.ilike(f'%{query}%')) |
            (Prediction.race_ethnicity.ilike(f'%{query}%')) |
            (Prediction.parental_level_of_education.ilike(f'%{query}%')) |
            (Prediction.lunch.ilike(f'%{query}%')) |
            (Prediction.test_preparation_course.ilike(f'%{query}%'))
        )
    
    predictions = search_query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('dashboard.html',
                           predictions=predictions,
                           search_query=query,
                           total_predictions=Prediction.query.count(),
                           avg_score=round(db.session.query(db.func.avg(Prediction.predicted_math_score)).scalar() or 0, 2),
                           max_score=db.session.query(db.func.max(Prediction.predicted_math_score)).scalar(),
                           min_score=db.session.query(db.func.min(Prediction.predicted_math_score)).scalar(),
                           gender_dist=db.session.query(Prediction.gender, db.func.count(Prediction.id)).group_by(Prediction.gender).all(),
                           test_prep_dist=db.session.query(Prediction.test_preparation_course, db.func.count(Prediction.id)).group_by(Prediction.test_preparation_course).all())

# Export predictions to CSV
@app.route('/export')
def export_predictions():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Gender', 'Race/Ethnicity', 'Parent Education', 'Lunch', 
                     'Test Prep Course', 'Reading Score', 'Writing Score', 
                     'Predicted Math Score', 'Created At'])
    
    # Write data
    for p in predictions:
        writer.writerow([p.id, p.gender, p.race_ethnicity, p.parental_level_of_education,
                        p.lunch, p.test_preparation_course, p.reading_score, 
                        p.writting_score, p.predicted_math_score, p.created_at.strftime('%Y-%m-%d %H:%M:%S')])
    
    # Create response
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=student_predictions.csv"}
    )

# Advanced statistics endpoint
@app.route('/api/advanced-stats')
def advanced_statistics():
    # Get all predictions for analysis
    predictions = Prediction.query.all()
    
    if not predictions:
        return jsonify({
            'total_predictions': 0,
            'message': 'No predictions available'
        })
    
    # Calculate various statistics
    total = len(predictions)
    
    # Average scores
    avg_math = round(np.mean([p.predicted_math_score for p in predictions]), 2)
    avg_reading = round(np.mean([p.reading_score for p in predictions]), 2)
    avg_writing = round(np.mean([p.writting_score for p in predictions]), 2)
    
    # Score ranges
    math_scores = [p.predicted_math_score for p in predictions]
    reading_scores = [p.reading_score for p in predictions]
    writing_scores = [p.writting_score for p in predictions]
    
    # Statistics by gender
    gender_stats = {}
    for gender in ['male', 'female']:
        gender_preds = [p for p in predictions if p.gender == gender]
        if gender_preds:
            gender_stats[gender] = {
                'count': len(gender_preds),
                'avg_math': round(np.mean([p.predicted_math_score for p in gender_preds]), 2),
                'avg_reading': round(np.mean([p.reading_score for p in gender_preds]), 2),
                'avg_writing': round(np.mean([p.writting_score for p in gender_preds]), 2)
            }
    
    # Statistics by test preparation
    test_prep_stats = {}
    for prep in ['none', 'completed']:
        prep_preds = [p for p in predictions if p.test_preparation_course == prep]
        if prep_preds:
            test_prep_stats[prep] = {
                'count': len(prep_preds),
                'avg_math': round(np.mean([p.predicted_math_score for p in prep_preds]), 2),
                'avg_reading': round(np.mean([p.reading_score for p in prep_preds]), 2),
                'avg_writing': round(np.mean([p.writting_score for p in prep_preds]), 2)
            }
    
    # Statistics by parental education
    education_stats = {}
    for edu in ['high school', 'some high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]:
        edu_preds = [p for p in predictions if p.parental_level_of_education == edu]
        if edu_preds:
            education_stats[edu] = {
                'count': len(edu_preds),
                'avg_math': round(np.mean([p.predicted_math_score for p in edu_preds]), 2)
            }
    
    # Lunch type statistics
    lunch_stats = {}
    for lunch_type in ['free/reduced', 'standard']:
        lunch_preds = [p for p in predictions if p.lunch == lunch_type]
        if lunch_preds:
            lunch_stats[lunch_type] = {
                'count': len(lunch_preds),
                'avg_math': round(np.mean([p.predicted_math_score for p in lunch_preds]), 2)
            }
    
    # Correlation between reading and math
    if len(reading_scores) > 1:
        correlation_reading_math = round(np.corrcoef(reading_scores, math_scores)[0, 1], 4)
    else:
        correlation_reading_math = 0
    
    if len(writing_scores) > 1:
        correlation_writing_math = round(np.corrcoef(writing_scores, math_scores)[0, 1], 4)
    else:
        correlation_writing_math = 0
    
    if len(reading_scores) > 1 and len(writing_scores) > 1:
        correlation_reading_writing = round(np.corrcoef(reading_scores, writing_scores)[0, 1], 4)
    else:
        correlation_reading_writing = 0
    
    return jsonify({
        'total_predictions': total,
        'averages': {
            'math': avg_math,
            'reading': avg_reading,
            'writing': avg_writing
        },
        'score_ranges': {
            'math': {'min': min(math_scores), 'max': max(math_scores)},
            'reading': {'min': min(reading_scores), 'max': max(reading_scores)},
            'writing': {'min': min(writing_scores), 'max': max(writing_scores)}
        },
        'by_gender': gender_stats,
        'by_test_preparation': test_prep_stats,
        'by_parental_education': education_stats,
        'by_lunch_type': lunch_stats,
        'correlations': {
            'reading_math': correlation_reading_math,
            'writing_math': correlation_writing_math,
            'reading_writing': correlation_reading_writing
        }
    })

# Trend analysis endpoint (predictions over time)
@app.route('/api/trends')
def trend_analysis():
    # Get predictions grouped by date
    from sqlalchemy import func
    
    daily_stats = db.session.query(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count'),
        func.avg(Prediction.predicted_math_score).label('avg_math')
    ).group_by(func.date(Prediction.created_at)).order_by(func.date(Prediction.created_at)).all()
    
    trends = []
    for stat in daily_stats:
        trends.append({
            'date': stat.date.strftime('%Y-%m-%d') if hasattr(stat.date, 'strftime') else str(stat.date),
            'count': stat.count,
            'avg_math': round(stat.avg_math, 2) if stat.avg_math else 0
        })
    
    return jsonify(trends)

# Filter predictions endpoint
@app.route('/filter')
def filter_predictions():
    gender = request.args.get('gender', '')
    ethnicity = request.args.get('ethnicity', '')
    test_prep = request.args.get('test_prep', '')
    min_score = request.args.get('min_score', type=float)
    max_score = request.args.get('max_score', type=float)
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    query = Prediction.query
    
    if gender:
        query = query.filter(Prediction.gender == gender)
    if ethnicity:
        query = query.filter(Prediction.race_ethnicity == ethnicity)
    if test_prep:
        query = query.filter(Prediction.test_preparation_course == test_prep)
    if min_score is not None:
        query = query.filter(Prediction.predicted_math_score >= min_score)
    if max_score is not None:
        query = query.filter(Prediction.predicted_math_score <= max_score)
    
    predictions = query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('dashboard.html',
                           predictions=predictions,
                           filter_gender=gender,
                           filter_ethnicity=ethnicity,
                           filter_test_prep=test_prep,
                           filter_min_score=min_score,
                           filter_max_score=max_score,
                           total_predictions=Prediction.query.count(),
                           avg_score=round(db.session.query(db.func.avg(Prediction.predicted_math_score)).scalar() or 0, 2),
                           max_score=db.session.query(db.func.max(Prediction.predicted_math_score)).scalar(),
                           min_score=db.session.query(db.func.min(Prediction.predicted_math_score)).scalar(),
                           gender_dist=db.session.query(Prediction.gender, db.func.count(Prediction.id)).group_by(Prediction.gender).all(),
                           test_prep_dist=db.session.query(Prediction.test_preparation_course, db.func.count(Prediction.id)).group_by(Prediction.test_preparation_course).all())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
