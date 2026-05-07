# Testing: Intelligent Column Exclusion v1.0

## Implementation Summary

✓ **Backend (main.py)**:
- Added `get_drop_suggestions()` function (line ~57): Detects high-cardinality and ID-like columns
- Added `/suggest-drops` endpoint (line ~175): Returns suggestions for suspicious columns

✓ **Frontend (index.html)**:
- Added `suggestions` state (line ~750)
- Added CSS for suggested columns (lines 313-337)
- Integrated `/suggest-drops` call in `handleFile()` (after /analyze)
- Added `handleAutoSuggest()` function (line ~850)
- Added suggestions panel with button to auto-exclude (lines 975-1006)
- Added visual badges to column cards (suggested-drop class)
- Updated `reset()` to clear suggestions

---

## Manual Testing Guide

### Prerequisites
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Test 1: Backend Endpoint
```bash
curl -F "file=@titanic_sample.csv" http://localhost:8000/suggest-drops
```

**Expected Response**:
```json
{
  "total_columns": 8,
  "suggestions": [
    {
      "column": "PassengerId",
      "reason": "Alta cardinalidad (891 únicos)",
      "confidence": 0.95,
      "type": "cardinality"
    },
    {
      "column": "Name",
      "reason": "Alta cardinalidad (891 únicos)",
      "confidence": 0.95,
      "type": "cardinality"
    }
  ],
  "message": "Detectadas 2 columnas potencialmente irrelevantes"
}
```

### Test 2: Frontend Flow (Manual in Browser)

1. **Open**: `file:///path/to/frontend/index.html`
2. **Upload CSV**: `backend/titanic_sample.csv`
3. **Verify**:
   - ✓ Yellow badges appear on PassengerId + Name columns
   - ✓ Suggestions panel shows below column grid
   - ✓ Panel displays: column name + reason + confidence %
   - ✓ Logs show: "⚠ 2 columnas sospechosas detectadas"

4. **Click "Excluir sugerencias" button**:
   - ✓ Badges change to "dropped" state (opacity: 0.4)
   - ✓ Log shows: "2 columnas marcadas para excluir"

5. **Select Target** (e.g., "Survived"):
   - ✓ Target highlighted in green

6. **Train**:
   - ✓ Model trains WITHOUT PassengerId + Name
   - ✓ Features should be: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
   - ✓ Accuracy should be similar or better (less noise)

7. **Verify Model Info** (GET `/model-info`):
   - Should show 7 features (not 8)

### Test 3: Edge Cases

**No suggestions**:
- Upload CSV with only meaningful columns → "Todas las columnas parecen relevantes"

**Manual override**:
- Upload CSV with suggestions
- Click column badge to toggle "dropped" state
- Deselect a suggested column → excluded from auto-exclude

**Very high cardinality**:
- Create CSV with column > 1000 unique values → should be flagged

---

## Heuristics Implemented

| Trigger | Confidence | Example |
|---------|------------|---------|
| >70% unique values | 0.95 | `PassengerId` (891 unique of 891 rows) |
| >1000 unique values | 0.95 | Large ID column |
| Contains "id" | 0.90 | `user_id`, `order_id` |
| Contains "uuid" | 0.90 | `uuid_field` |
| Contains "key" | 0.90 | `api_key` |
| Contains "codigo" | 0.90 | `codigo_postal` |

---

## Files Changed

- `backend/main.py`: +50 lines (function + endpoint)
- `frontend/index.html`: +100 lines (state, CSS, panel, logic)

---

## Known Limitations

1. **Heuristics are conservative**: Only high-cardinality + ID-like names
   - Can't detect semantic irrelevance (domain-specific)
2. **Cardinality threshold (0.7) hardcoded**: May need tuning for different datasets
3. **No statistical analysis**: Doesn't use correlation, entropy, etc.

---

## Next Steps (Future Enhancements)

- [ ] Add mutual information scoring
- [ ] Add correlation-based suggestions (duplicate features)
- [ ] Remember user preferences (skip suggestions for certain column names)
- [ ] Admin panel to tune thresholds
- [ ] A/B test: with/without suggestions → model performance
