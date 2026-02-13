import os

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB upload limit
UPLOAD_ALLOWED_EXTENSIONS = {"csv", "tsv", "xlsx", "xls", "json"}
TOP_N_CATEGORICAL = 10
TITLE_ID_UNIQUENESS_THRESHOLD = 0.95
TITLE_ID_MAX_CATEGORIES = 100
