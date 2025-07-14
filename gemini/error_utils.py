class ErrorUtils:
    @staticmethod
    def handle_analysis_error(error: Exception, context: str) -> str:
        error_msg = f"Error in {context}: {str(error)}"
        print(error_msg)
        return f"""
        Analysis Error
        -------------
        Context: {context}
        Error: {str(error)}
        Recommendation: Please try again with a different chart or contact support if the issue persists.
        """ 