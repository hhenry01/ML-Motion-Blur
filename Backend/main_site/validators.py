def validate_file_extension(value):
    import os
    from django.core.excepts import ValidationError
    ext = os.path.splitext(value.name)[1] # returns path + filename
    valid_extensions = ['.mp4', '.mov', '.wmv', '.avi']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension.')