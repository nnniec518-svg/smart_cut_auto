@echo off
chcp 65001 > nul
echo ========================================
echo 测试 BGE 模型加载
echo ========================================
echo.

cd /d "%~dp0"
python -c "from sentence_transformers import SentenceTransformer; import os; os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'models/sentence_transformers'; model = SentenceTransformer('BAAI/bge-large-zh-v1.5'); print('✅ BGE模型加载成功'); print(f'维度: {model.get_sentence_embedding_dimension()}')"

echo.
pause
