import os
from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from models import User
from forms import LoginForm, RegistrationForm, UserBasedRecommendationForm, ItemBasedRecommendationForm, DeepLearningRecommendationForm
from utils import get_dataset_stats, get_available_user_ids, get_available_product_ids
from recommenders import (
    get_user_based_recommendations,
    get_item_based_recommendations,
    get_deep_learning_recommendations
)

def register_routes(app):
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = RegistrationForm()
        if form.validate_on_submit():
            user = User(username=form.username.data, email=form.email.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html', form=form)
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.check_password(form.password.data):
                login_user(user)
                flash('Login successful!', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            else:
                flash('Invalid email or password.', 'danger')
        
        return render_template('login.html', form=form)
    
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('index'))
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        stats = get_dataset_stats()
        return render_template('dashboard.html', stats=stats)
    
    @app.route('/user_based', methods=['GET', 'POST'])
    @login_required
    def user_based():
        form = UserBasedRecommendationForm()
        recommendations = []
        user_exists = True
        
        if form.validate_on_submit():
            user_id = form.user_id.data
            try:
                recommendations = get_user_based_recommendations(user_id, top_n=5)
                if not recommendations:
                    flash(f"No recommendations found for user ID: {user_id}", "warning")
            except ValueError as e:
                user_exists = False
                flash(str(e), "danger")
        
        # Get some sample user IDs to help the user
        sample_user_ids = get_available_user_ids(limit=10)
        
        return render_template(
            'user_based.html', 
            form=form, 
            recommendations=recommendations,
            sample_user_ids=sample_user_ids,
            user_exists=user_exists
        )
    
    @app.route('/item_based', methods=['GET', 'POST'])
    @login_required
    def item_based():
        form = ItemBasedRecommendationForm()
        recommendations = []
        item_exists = True
        
        if form.validate_on_submit():
            product_id = form.product_id.data
            try:
                recommendations = get_item_based_recommendations(product_id, top_n=5)
                if not recommendations:
                    flash(f"No recommendations found for product ID: {product_id}", "warning")
            except ValueError as e:
                item_exists = False
                flash(str(e), "danger")
        
        # Get some sample product IDs to help the user
        sample_product_ids = get_available_product_ids(limit=10)
        
        return render_template(
            'item_based.html', 
            form=form, 
            recommendations=recommendations,
            sample_product_ids=sample_product_ids,
            item_exists=item_exists
        )
    
    @app.route('/deep_learning', methods=['GET', 'POST'])
    @login_required
    def deep_learning():
        form = DeepLearningRecommendationForm()
        recommendations = []
        user_exists = True
        
        if form.validate_on_submit():
            user_id = form.user_id.data
            try:
                recommendations = get_deep_learning_recommendations(user_id, top_n=5)
                if not recommendations:
                    flash(f"No recommendations found for user ID: {user_id}", "warning")
            except ValueError as e:
                user_exists = False
                flash(str(e), "danger")
        
        # Get some sample user IDs to help the user
        sample_user_ids = get_available_user_ids(limit=10)
        
        return render_template(
            'deep_learning.html', 
            form=form, 
            recommendations=recommendations,
            sample_user_ids=sample_user_ids,
            user_exists=user_exists
        )
    
    @app.route('/dataset_stats')
    @login_required
    def dataset_stats():
        stats = get_dataset_stats()
        return render_template('dataset_stats.html', stats=stats)
    
    @app.route('/about')
    def about():
        return render_template('about.html')
    
    @app.route('/api/dataset_stats')
    @login_required
    def api_dataset_stats():
        stats = get_dataset_stats()
        return jsonify(stats)
