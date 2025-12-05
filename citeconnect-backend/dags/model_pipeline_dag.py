"""
CiteConnect Model Development Pipeline DAG
Validation tasks - one by one
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
import sys
from pathlib import Path
import asyncio

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_db_connection(**context):
    """Task 1: Test database connection"""
    print("="*80, flush=True)
    print("TASK 1: TEST DATABASE CONNECTION", flush=True)
    print("="*80, flush=True)
    
    try:
        from dags.airflow_db_helper import SimpleDB
        db = SimpleDB()
        
        async def connect_test():
            await db.connect()
            print("✅ Connected!", flush=True)
            result = await db.fetchval("SELECT 1")
            print(f"✅ Query result: {result}", flush=True)
            await db.disconnect()
            print("✅ Disconnected", flush=True)
            return "success"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(connect_test())
            print(f"\n✅ Task 1 completed: {result}", flush=True)
            return "success"
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    except Exception as e:
        print(f"\n❌ Task 1 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return "error"


def check_papers(**context):
    """Task 2: Check if we have papers"""
    print("="*80, flush=True)
    print("TASK 2: CHECK PAPERS", flush=True)
    print("="*80, flush=True)
    
    try:
        from dags.airflow_db_helper import SimpleDB
        db = SimpleDB()
        
        async def check():
            await db.connect()
            print("✅ Connected", flush=True)
            
            # Check papers count
            count = await db.fetchval("SELECT COUNT(*) FROM papers")
            print(f"📊 Total papers: {count}", flush=True)
            
            if count == 0:
                print("❌ No papers found!", flush=True)
                await db.disconnect()
                return "error"
            
            print(f"✅ Papers check passed: {count} papers", flush=True)
            await db.disconnect()
            return "success"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(check())
            print(f"\n✅ Task 2 completed: {result}", flush=True)
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    except Exception as e:
        print(f"\n❌ Task 2 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return "error"


def check_embeddings(**context):
    """Task 3: Check if all papers have embeddings"""
    print("="*80, flush=True)
    print("TASK 3: CHECK EMBEDDINGS", flush=True)
    print("="*80, flush=True)
    
    try:
        from dags.airflow_db_helper import SimpleDB
        db = SimpleDB()
        
        async def check():
            await db.connect()
            print("✅ Connected", flush=True)
            
            # Get total papers
            total_papers = await db.fetchval("SELECT COUNT(*) FROM papers")
            print(f"📊 Total papers: {total_papers}", flush=True)
            
            # Check SPECTER2 embeddings
            specter_count = await db.fetchval(
                "SELECT COUNT(DISTINCT paper_id) FROM paper_embeddings_specter WHERE embedding IS NOT NULL"
            )
            specter_coverage = (specter_count / total_papers * 100) if total_papers > 0 else 0
            print(f"📊 SPECTER2: {specter_count}/{total_papers} ({specter_coverage:.1f}%)", flush=True)
            
            # Check MiniLM embeddings
            minilm_count = await db.fetchval(
                "SELECT COUNT(DISTINCT paper_id) FROM paper_embeddings_minilm WHERE embedding IS NOT NULL"
            )
            minilm_coverage = (minilm_count / total_papers * 100) if total_papers > 0 else 0
            print(f"📊 MiniLM: {minilm_count}/{total_papers} ({minilm_coverage:.1f}%)", flush=True)
            
            # Check if both have good coverage
            if specter_coverage >= 90 and minilm_coverage >= 90:
                print("✅ Embeddings check passed: Both models have >= 90% coverage", flush=True)
                await db.disconnect()
                return "success"
            else:
                print(f"⚠️  Low coverage: SPECTER={specter_coverage:.1f}%, MiniLM={minilm_coverage:.1f}%", flush=True)
                await db.disconnect()
                return "error"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(check())
            print(f"\n✅ Task 3 completed: {result}", flush=True)
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    except Exception as e:
        print(f"\n❌ Task 3 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return "error"


def check_cold_start_users(**context):
    """Task 4: Check for cold-start users (< 10 interactions)"""
    print("="*80, flush=True)
    print("TASK 4: CHECK COLD-START USERS", flush=True)
    print("="*80, flush=True)
    
    try:
        from dags.airflow_db_helper import SimpleDB
        db = SimpleDB()
        
        async def check():
            await db.connect()
            print("✅ Connected", flush=True)
            
            # Count cold-start users (users with < 10 interactions)
            cold_start_count = await db.fetchval("""
                SELECT COUNT(DISTINCT user_id)
                FROM (
                    SELECT u.user_id, COUNT(ui.interaction_id) as interaction_count
                    FROM users u
                    LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
                    WHERE u.is_active = true
                    GROUP BY u.user_id
                ) subq
                WHERE interaction_count < 10
            """)
            
            print(f"📊 Cold-start users (< 10 interactions): {cold_start_count}", flush=True)
            
            if cold_start_count >= 20:
                print(f"✅ Cold-start users check passed: {cold_start_count} users (need >= 20)", flush=True)
                await db.disconnect()
                return "success"
            else:
                print(f"❌ Insufficient cold-start users: {cold_start_count} (need >= 20)", flush=True)
                await db.disconnect()
                return "error"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(check())
            print(f"\n✅ Task 4 completed: {result}", flush=True)
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    except Exception as e:
        print(f"\n❌ Task 4 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return "error"


def check_ground_truth(**context):
    """Task 5: Check ground truth papers and relationships"""
    print("="*80, flush=True)
    print("TASK 5: CHECK GROUND TRUTH", flush=True)
    print("="*80, flush=True)
    
    try:
        from dags.airflow_db_helper import SimpleDB
        db = SimpleDB()
        
        async def check():
            await db.connect()
            print("✅ Connected", flush=True)
            
            # Check ground_truth_papers
            gt_papers_count = await db.fetchval("SELECT COUNT(*) FROM ground_truth_papers")
            print(f"📊 Ground truth papers: {gt_papers_count}", flush=True)
            
            # Check ground_truth_relationships
            try:
                gt_rels_count = await db.fetchval("SELECT COUNT(*) FROM ground_truth_relationships")
                print(f"📊 Ground truth relationships: {gt_rels_count}", flush=True)
            except Exception as e:
                print(f"⚠️  Could not check relationships table: {str(e)}", flush=True)
                gt_rels_count = 0
            
            if gt_papers_count > 0:
                print(f"✅ Ground truth check passed: {gt_papers_count} papers, {gt_rels_count} relationships", flush=True)
                await db.disconnect()
                return "success"
            else:
                print("❌ No ground truth papers found!", flush=True)
                await db.disconnect()
                return "error"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(check())
            print(f"\n✅ Task 5 completed: {result}", flush=True)
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    except Exception as e:
        print(f"\n❌ Task 5 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return "error"


def test_recommendations(**context):
    """Task 6: Test recommendations for ONE cold-start user"""
    print("="*80, flush=True)
    print("TASK 6: TEST RECOMMENDATIONS (ONE USER)", flush=True)
    print("="*80, flush=True)
    
    try:
        # Import services - use app.db.connection for full functionality
        from app.db.connection import db
        from app.services.recommendation_service import RecommendationService
        from app.services.evaluation_service import EvaluationService
        
        async def test():
            await db.connect()
            print("✅ Connected to database", flush=True)
            
            # Step 1: Get ONE cold-start user (< 10 interactions)
            print("\n📋 Step 1: Finding one cold-start user...", flush=True)
            user_row = await db.fetchrow("""
                SELECT u.user_id, COUNT(ui.interaction_id) as interaction_count
                FROM users u
                LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
                WHERE u.is_active = true
                GROUP BY u.user_id
                HAVING COUNT(ui.interaction_id) < 10
                LIMIT 1
            """)
            
            if not user_row:
                print("❌ No cold-start users found!", flush=True)
                await db.disconnect()
                return "error"
            
            user_id = user_row['user_id']
            interaction_count = user_row['interaction_count']
            print(f"✅ Found user_id={user_id} with {interaction_count} interactions", flush=True)
            
            # Step 2: Generate 10 recommendations
            print(f"\n📋 Step 2: Generating 10 recommendations for user {user_id}...", flush=True)
            rec_service = RecommendationService(db)
            
            recommendations = await rec_service.generate_cold_start_recommendations(
                user_id=user_id,
                count=10,
                model='minilm'  # Using MiniLM model
            )
            
            rec_count = len(recommendations.get('papers', []))
            print(f"✅ Generated {rec_count} recommendations", flush=True)
            
            if rec_count == 0:
                print("❌ No recommendations generated!", flush=True)
                await db.disconnect()
                return "error"
            
            # Print first few paper IDs
            paper_ids = [p.get('paper_id', 'N/A') for p in recommendations.get('papers', [])[:3]]
            print(f"   Sample paper IDs: {paper_ids}", flush=True)
            
            # Step 3: Evaluate recommendations (WITHOUT storing to DB for testing)
            print(f"\n📋 Step 3: Evaluating recommendations...", flush=True)
            eval_service = EvaluationService(db)
            
            evaluation = await eval_service.evaluate_cold_start_recommendations(
                user_id=user_id,
                recommendations=recommendations['papers'],
                model='minilm',
                store_result=False  # ⚠️ NOT storing to DB - just testing!
            )
            
            # Step 4: Print scores
            print("\n" + "="*60, flush=True)
            print("EVALUATION RESULTS", flush=True)
            print("="*60, flush=True)
            print(f"User ID: {user_id}", flush=True)
            print(f"Recommendations: {rec_count}", flush=True)
            print(f"\n📊 Scores:", flush=True)
            print(f"   Profile Alignment: {evaluation['profile_alignment']:.4f}", flush=True)
            print(f"   Ground Truth Quality: {evaluation['ground_truth_quality']:.4f}", flush=True)
            print(f"   Combined Score: {evaluation['combined_score']:.4f}", flush=True)
            print(f"   Passes Threshold (≥0.60): {'✅ YES' if evaluation['passes_threshold'] else '❌ NO'}", flush=True)
            print(f"\n📋 Thresholds:", flush=True)
            thresholds = evaluation.get('thresholds', {})
            print(f"   Profile Alignment: ≥{thresholds.get('profile_alignment', 'N/A')}", flush=True)
            print(f"   Ground Truth Quality: ≥{thresholds.get('ground_truth_quality', 'N/A')}", flush=True)
            print(f"   Combined Score: ≥{thresholds.get('combined_score', 'N/A')}", flush=True)
            
            await db.disconnect()
            print("\n✅ Test completed successfully!", flush=True)
            return "success"
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test())
            print(f"\n✅ Task 6 completed: {result}", flush=True)
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    except Exception as e:
        print(f"\n❌ Task 6 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return "error"


def detect_bias(**context):
    """
    Task 7: Detect bias in cold-start recommendations across ALL users
    
    READ-ONLY: Only reads from database (SELECT queries only)
    - Reads from: cold_start_evaluations, user_profiles_extended
    - Writes to: JSON files only (bias_report_cold_start_before.json)
    - NO database modifications
    """
    print("="*80, flush=True)
    print("TASK 7: DETECT BIAS (READ-ONLY)", flush=True)
    print("="*80, flush=True)
    
    try:
        from app.db.connection import db
        from pathlib import Path
        from collections import defaultdict
        from datetime import datetime
        import json
        import numpy as np
        
        # Bias detection constants (from bias_slicing_cold_start.py)
        BIAS_DISPARITY_THRESHOLD = 0.15  # 15 percentage points
        MIN_USERS_PER_SLICE = 2
        BASE_DIR = Path(__file__).parent.parent
        BIAS_REPORT_PATH = BASE_DIR / "bias_report_cold_start_before.json"
        
        # Inline functions from bias_slicing_cold_start.py
        async def load_joined_data(db_conn):
            """Load evaluation data joined with user profiles (SELECT only)"""
            query = """
            SELECT
                c.user_id,
                c.embedding_model,
                c.profile_alignment,
                c.ground_truth_quality,
                c.combined_score,
                c.recommendation_count,
                c.evaluation_timestamp,
                p.primary_domain,
                p.research_stage,
                p.reading_level,
                p.years_experience
            FROM cold_start_evaluations c
            JOIN user_profiles_extended p
              ON c.user_id = p.user_id
            ORDER BY c.user_id;
            """
            rows = await db_conn.fetch(query)
            return [dict(r) for r in rows]
        
        def aggregate_by_slice(rows):
            """Group rows by slice dimensions and compute metrics"""
            slice_fields = ["primary_domain", "research_stage", "reading_level"]
            raw_data = {field: defaultdict(list) for field in slice_fields}
            
            for row in rows:
                for field in slice_fields:
                    value = row.get(field) or "unknown"
                    raw_data[field][value].append(row)
            
            slice_metrics = {}
            for field in slice_fields:
                field_metrics = {}
                for value, group in raw_data[field].items():
                    if len(group) == 0:
                        continue
                    
                    combined_scores = [g["combined_score"] for g in group if g["combined_score"] is not None]
                    profile_alignments = [g["profile_alignment"] for g in group if g["profile_alignment"] is not None]
                    ground_truth_qualities = [g["ground_truth_quality"] for g in group if g["ground_truth_quality"] is not None]
                    
                    if not combined_scores:
                        continue
                    
                    field_metrics[value] = {
                        "user_count": len(group),
                        "mean_combined_score": float(np.mean(combined_scores)),
                        "std_combined_score": float(np.std(combined_scores)),
                        "mean_profile_alignment": float(np.mean(profile_alignments)) if profile_alignments else None,
                        "mean_ground_truth_quality": float(np.mean(ground_truth_qualities)) if ground_truth_qualities else None,
                    }
                slice_metrics[field] = field_metrics
            return slice_metrics
        
        def detect_bias_findings(slice_metrics):
            """Detect bias when disparity > threshold"""
            bias_findings = []
            metrics_to_check = ["mean_combined_score", "mean_profile_alignment", "mean_ground_truth_quality"]
            
            for field, slices in slice_metrics.items():
                valid_slices = {
                    name: m for name, m in slices.items()
                    if m.get("user_count", 0) >= MIN_USERS_PER_SLICE
                }
                if len(valid_slices) < 2:
                    continue
                
                for metric_name in metrics_to_check:
                    values = []
                    for slice_name, m in valid_slices.items():
                        value = m.get(metric_name)
                        if value is not None:
                            values.append((slice_name, value))
                    
                    if len(values) < 2:
                        continue
                    
                    best_slice, best_val = max(values, key=lambda x: x[1])
                    worst_slice, worst_val = min(values, key=lambda x: x[1])
                    
                    if best_val == 0:
                        continue
                    
                    disparity = (best_val - worst_val) / best_val
                    
                    if disparity > BIAS_DISPARITY_THRESHOLD:
                        bias_findings.append({
                            "field": field,
                            "metric": metric_name,
                            "best_slice": best_slice,
                            "best_value": best_val,
                            "worst_slice": worst_slice,
                            "worst_value": worst_val,
                            "disparity": disparity,
                        })
            return bias_findings
        
        def build_report(rows, slice_metrics, bias_findings):
            """Package everything into a JSON-serializable dict"""
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "total_users_in_eval": len({r["user_id"] for r in rows}),
                "total_eval_rows": len(rows),
                "slices_analyzed": list(slice_metrics.keys()),
                "slice_metrics": slice_metrics,
                "bias_findings": bias_findings,
            }
        
        def save_report(report, path):
            """Save report to JSON file (file system only, not database)"""
            path.write_text(json.dumps(report, indent=2))
            print(f"✅ Bias report saved to {path.resolve()}", flush=True)
        
        async def run_detection():
            await db.connect()
            print("✅ Connected to database", flush=True)
            print("⚠️  READ-ONLY MODE: Only SELECT queries, no database writes", flush=True)
            
            # Step 1: Load data using existing function (SELECT only)
            print("\n📋 Step 1: Loading evaluation data from ALL users...", flush=True)
            rows = await load_joined_data(db)
            print(f"✅ Loaded {len(rows)} evaluation rows", flush=True)
            
            if not rows:
                print("⚠️  No evaluation data found in cold_start_evaluations", flush=True)
                await db.disconnect()
                return {"status": "warning", "bias_findings": [], "report_path": None, "bias_count": 0}
            
            # Step 2: Aggregate by slices using existing function
            print("\n📋 Step 2: Aggregating by slices (domain, research_stage, reading_level)...", flush=True)
            slice_metrics = aggregate_by_slice(rows)
            print(f"✅ Aggregated metrics for {len(slice_metrics)} slice types", flush=True)
            
            # Step 3: Detect bias using existing function
            print("\n📋 Step 3: Detecting bias (threshold: 15% disparity)...", flush=True)
            bias_findings = detect_bias_findings(slice_metrics)
            print(f"✅ Found {len(bias_findings)} bias findings", flush=True)
            
            # Step 4: Build and save report using existing functions (file write only, not DB)
            print("\n📋 Step 4: Saving bias report to JSON file...", flush=True)
            report = build_report(rows, slice_metrics, bias_findings)
            save_report(report, BIAS_REPORT_PATH)
            
            # Print summary
            print("\n" + "="*60, flush=True)
            print("BIAS DETECTION SUMMARY", flush=True)
            print("="*60, flush=True)
            if not bias_findings:
                print("✅ No significant bias detected (all disparities < 15%)", flush=True)
            else:
                print(f"⚠️  {len(bias_findings)} bias finding(s) detected:", flush=True)
                for b in bias_findings:
                    print(f"   - {b['field']}.{b['metric']}: {b['best_slice']} ({b['best_value']:.3f}) vs {b['worst_slice']} ({b['worst_value']:.3f}) - disparity: {b['disparity']:.2%}", flush=True)
            
            await db.disconnect()
            
            return {
                "status": "success" if not bias_findings else "bias_detected",
                "bias_findings": bias_findings,
                "report_path": str(BIAS_REPORT_PATH),
                "bias_count": len(bias_findings),
                "generated_at": report.get("generated_at")
            }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_detection())
            print(f"\n✅ Task 7 completed: {result['status']}", flush=True)
            return result
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    except Exception as e:
        print(f"\n❌ Task 7 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return {"status": "error", "bias_findings": [], "report_path": None, "bias_count": 0}


def send_bias_alert(**context):
    """
    Task 8: Send email alert if bias is detected
    
    READ-ONLY: Only reads from XCom (previous task results)
    - Reads from: XCom (bias detection results)
    - Writes to: Email only (no database operations)
    - NO database modifications
    """
    print("="*80, flush=True)
    print("TASK 8: SEND BIAS ALERT (READ-ONLY)", flush=True)
    print("="*80, flush=True)
    
    try:
        import os
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Get bias detection results from previous task (XCom - no DB read)
        ti = context['ti']
        bias_result = ti.xcom_pull(task_ids='detect_bias')
        
        if not bias_result or bias_result.get('status') != 'bias_detected':
            print("✅ No bias detected - no alert needed", flush=True)
            return "success"
        
        bias_findings = bias_result.get('bias_findings', [])
        bias_count = bias_result.get('bias_count', 0)
        report_path = bias_result.get('report_path')
        
        print(f"⚠️  Bias detected: {bias_count} finding(s)", flush=True)
        print("⚠️  READ-ONLY MODE: No database operations", flush=True)
        
        # Email configuration from environment variables
        smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_user = os.getenv('SMTP_USER', '')
        smtp_password = os.getenv('SMTP_PASSWORD', '')
        alert_email_to = os.getenv('BIAS_ALERT_EMAIL', 'admin@citeconnect.io')
        alert_email_from = os.getenv('SMTP_USER', smtp_user)
        
        if not smtp_user or not smtp_password:
            print("⚠️  SMTP credentials not configured - skipping email", flush=True)
            print("   Set SMTP_USER, SMTP_PASSWORD, and BIAS_ALERT_EMAIL environment variables", flush=True)
            return "warning"
        
        # Build email content
        print("\n📋 Building email content...", flush=True)
        
        subject = f"⚠️ CiteConnect Bias Alert: {bias_count} Bias Finding(s) Detected"
        
        # Create email body
        body_text = f"""
CiteConnect Bias Detection Alert

Bias has been detected in the cold-start recommendation system.

Summary:
- Total bias findings: {bias_count}
- Detection timestamp: {bias_result.get('generated_at', 'N/A')}

Bias Findings:
"""
        
        for i, finding in enumerate(bias_findings, 1):
            body_text += f"""
{i}. {finding['field']}.{finding['metric']}
   - Best slice: {finding['best_slice']} (score: {finding['best_value']:.4f})
   - Worst slice: {finding['worst_slice']} (score: {finding['worst_value']:.4f})
   - Disparity: {finding['disparity']:.2%}
"""
        
        body_text += f"""

Full report available at: {report_path}

Please review the bias findings and consider updating the bias mitigation configuration.
"""
        
        # Create HTML version
        body_html = f"""
<html>
<head></head>
<body>
<h2>CiteConnect Bias Detection Alert</h2>
<p>Bias has been detected in the cold-start recommendation system.</p>

<h3>Summary</h3>
<ul>
<li><strong>Total bias findings:</strong> {bias_count}</li>
<li><strong>Detection timestamp:</strong> {bias_result.get('generated_at', 'N/A')}</li>
</ul>

<h3>Bias Findings</h3>
<ol>
"""
        
        for finding in bias_findings:
            body_html += f"""
<li>
<strong>{finding['field']}.{finding['metric']}</strong><br>
- Best slice: <strong>{finding['best_slice']}</strong> (score: {finding['best_value']:.4f})<br>
- Worst slice: <strong>{finding['worst_slice']}</strong> (score: {finding['worst_value']:.4f})<br>
- Disparity: <strong>{finding['disparity']:.2%}</strong>
</li>
"""
        
        body_html += f"""
</ol>

<p>Full report available at: <code>{report_path}</code></p>
<p>Please review the bias findings and consider updating the bias mitigation configuration.</p>
</body>
</html>
"""
        
        # Send email
        print(f"\n📋 Sending email to {alert_email_to}...", flush=True)
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = alert_email_from
        msg['To'] = alert_email_to
        
        part1 = MIMEText(body_text, 'plain')
        part2 = MIMEText(body_html, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            print(f"✅ Email sent successfully to {alert_email_to}", flush=True)
            return "success"
        
        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return "error"
    
    except Exception as e:
        print(f"\n❌ Task 8 error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return "error"


# DAG Definition
with DAG(
    'citeconnect_model_pipeline',
    start_date=days_ago(1),
    schedule=None,
    catchup=False,
    tags=['citeconnect']
) as dag:
    
    # Task 1: Test DB connection
    task1 = PythonOperator(
        task_id='test_db_connection',
        python_callable=test_db_connection
    )
    
    # Task 2: Check papers
    task2 = PythonOperator(
        task_id='check_papers',
        python_callable=check_papers
    )
    
    # Task 3: Check embeddings
    task3 = PythonOperator(
        task_id='check_embeddings',
        python_callable=check_embeddings
    )
    
    # Task 4: Check cold-start users
    task4 = PythonOperator(
        task_id='check_cold_start_users',
        python_callable=check_cold_start_users
    )
    
    # Task 5: Check ground truth
    task5 = PythonOperator(
        task_id='check_ground_truth',
        python_callable=check_ground_truth
    )
    
    # Task 6: Test recommendations for one cold-start user
    task6 = PythonOperator(
        task_id='test_recommendations',
        python_callable=test_recommendations
    )
    
    # Task 7: Detect bias across ALL users (READ-ONLY)
    task7 = PythonOperator(
        task_id='detect_bias',
        python_callable=detect_bias
    )
    
    # Task 8: Send bias alert via email (READ-ONLY)
    task8 = PythonOperator(
        task_id='send_bias_alert',
        python_callable=send_bias_alert
    )
    
    # Set dependencies: sequential execution
    task1 >> task2 >> task3 >> task4 >> task5 >> task6 >> task7 >> task8
