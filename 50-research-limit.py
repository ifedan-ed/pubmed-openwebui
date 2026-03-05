"""
title: Smart PubMed Research Assistant
author: Research Assistant
version: 5.5.0
date: 2025-01-01
license: MIT
description: Intelligent PubMed research assistant with ABSTRACT fetching. Passes full abstracts to the AI model for synthesis and analysis. Uses PubMed's Automatic Term Mapping — no dumb word splitting.
"""

import requests
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from pydantic import Field


class Tools:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self._cache = {}
        self._last_results = []
        self._last_query = ""

    # ================================================================
    # MAIN SEARCH
    # ================================================================

    def search_pubmed(
        self,
        query: str = Field(
            ...,
            description="Your research question in plain English. Examples: 'current guidelines for management of gastric reflux in children', 'low back pain treatment in elderly', 'ECMO outcomes in neonates'",
        ),
        max_results: int = Field(
            10,
            description="How many relevant articles you want (1-50)",
        ),
        include_abstracts: bool = Field(
            True,
            description="Include full abstracts (allows AI to synthesize findings). Set False for faster search without abstracts.",
        ),
    ) -> str:
        """
        Intelligent PubMed search with full abstract retrieval.
        The AI can read and synthesize the abstracts to answer your question.
        Just ask naturally.
        """
        try:
            max_results = self._safe_int(max_results, 10, 1, 50)
            query = str(query).strip()
            if not query:
                return "Please ask me a research question."

            # Handle include_abstracts safely
            if isinstance(include_abstracts, str):
                include_abstracts = include_abstracts.lower() not in (
                    "false",
                    "no",
                    "0",
                )
            elif not isinstance(include_abstracts, bool):
                include_abstracts = True

            # PHASE 1: Let PubMed analyze the query
            analysis = self._analyze_via_pubmed(query)

            # PHASE 2: Detect query type
            query_type = self._detect_query_type(query)

            # PHASE 3: Iterative search
            all_articles, search_log = self._iterative_search(
                query, analysis, query_type, max_results
            )

            # PHASE 4: Fetch abstracts for all collected articles
            if include_abstracts and all_articles:
                pmids = [a["pmid"] for a in all_articles]
                abstracts = self._fetch_abstracts(pmids)

                for article in all_articles:
                    article["abstract"] = abstracts.get(article["pmid"], "")

            # PHASE 5: Score relevance (now using abstracts too)
            scored = self._score_relevance(all_articles, query, query_type)
            top = scored[:max_results]

            # PHASE 6: Store
            self._last_results = top
            self._last_query = query

            # PHASE 7: Format
            if not top:
                return self._format_no_results(query, analysis, search_log)

            return self._format_results(
                query,
                analysis,
                query_type,
                search_log,
                top,
                len(all_articles),
                include_abstracts,
            )

        except Exception as e:
            return f"Search error: {str(e)}\n\nTry rephrasing your question."

    # ================================================================
    # GET RESULTS
    # ================================================================

    def get_results(
        self,
        format: str = Field(
            "list",
            description="'list' (references), 'ris' (Zotero), 'summary' (synthesis with abstracts), 'abstracts' (just abstracts), 'detailed' (everything)",
        ),
    ) -> str:
        """Get last search results in different formats."""
        try:
            fmt = str(format).strip().lower() if format else "list"
            if not self._last_results:
                return "No stored results. Run `search_pubmed` first."

            if fmt == "ris":
                return self._export_ris()
            elif fmt == "summary":
                return self._synthesize()
            elif fmt == "abstracts":
                return self._format_abstracts_only()
            elif fmt == "detailed":
                return self._format_detailed()
            else:
                return self._format_reference_list()
        except Exception as e:
            return f"Error: {str(e)}"

    # ================================================================
    # PICO SEARCH
    # ================================================================

    def pico_search(
        self,
        population: str = Field(..., description="Who?"),
        intervention: str = Field("", description="What treatment/exposure?"),
        comparison: str = Field("", description="Versus what?"),
        outcome: str = Field("", description="What outcome?"),
        max_results: int = Field(15, description="How many (1-50)"),
    ) -> str:
        """PICO framework search with abstracts for synthesis."""
        try:
            max_results = self._safe_int(max_results, 15, 1, 50)

            pico = {}
            for label, val in [
                ("Population", population),
                ("Intervention", intervention),
                ("Comparison", comparison),
                ("Outcome", outcome),
            ]:
                val = str(val).strip() if val else ""
                if val:
                    pico[label] = val

            if not pico:
                return "Please provide at least a Population."

            pico_analysis = {}
            for comp, text in pico.items():
                pico_analysis[comp] = self._analyze_via_pubmed(text)

            # Build and run strategies
            all_articles, search_log = self._pico_iterative_search(
                pico, pico_analysis, max_results
            )

            # Fetch abstracts
            if all_articles:
                pmids = [a["pmid"] for a in all_articles]
                abstracts = self._fetch_abstracts(pmids)
                for a in all_articles:
                    a["abstract"] = abstracts.get(a["pmid"], "")

            combined_query = " ".join(pico.values())
            scored = self._score_relevance(
                all_articles, combined_query, self._detect_query_type(combined_query)
            )
            top = scored[:max_results]

            self._last_results = top
            self._last_query = f"PICO: {pico}"

            # Format
            md = "# 🔬 PICO Search Results\n\n"
            md += "## Framework\n\n"
            md += "| Component | Your Input | PubMed Mapped To |\n"
            md += "|-----------|-----------|------------------|\n"
            for comp, text in pico.items():
                a = pico_analysis[comp]
                mapped = ", ".join(a.get("mesh_found", [])[:3]) or a.get(
                    "cleaned_query", text
                )
                md += f"| **{comp}** | {text} | {mapped} |\n"
            md += "\n"

            for s in search_log:
                icon = "✅" if s["found"] > 0 else "⭕"
                md += f"{icon} **{s['name']}** → {s['found']} results  \n"

            md += "\n"

            if top:
                md += f"## Top {len(top)} Results\n\n"
                md += self._format_article_list(top, show_abstracts=True)
            else:
                md += "**No results.** Try broader terms.\n"

            md += self._format_next_steps()
            return md

        except Exception as e:
            return f"PICO error: {str(e)}"

    # ================================================================
    # MESH FINDER
    # ================================================================

    def find_mesh(
        self,
        topic: str = Field(..., description="Any medical topic"),
    ) -> str:
        """Find MeSH terms using PubMed's own term mapping."""
        try:
            topic = str(topic).strip()
            analysis = self._analyze_via_pubmed(topic)

            md = f"# 🏷️ MeSH Terms for: {topic}\n\n"

            if analysis["mesh_found"]:
                md += "| MeSH Term | Search Syntax |\n"
                md += "|-----------|---------------|\n"
                for t in analysis["mesh_found"]:
                    md += f'| {t} | `"{t}"[MeSH]` |\n'
                md += "\n"

            if analysis["query_translation"]:
                md += f"**PubMed Translation:**\n```\n{analysis['query_translation']}\n```\n\n"

            return md

        except Exception as e:
            return f"MeSH error: {str(e)}"

    # ================================================================
    # CORE: ABSTRACT FETCHING
    # ================================================================

    def _fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """
        Fetch full abstracts from PubMed for a list of PMIDs.
        Returns {pmid: abstract_text}

        This is the KEY function that enables AI synthesis.
        """

        abstracts = {}
        if not pmids:
            return abstracts

        # Process in batches of 20 to avoid API limits
        batch_size = 20
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]

            try:
                resp = requests.get(
                    f"{self.base_url}/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "rettype": "xml",
                        "retmode": "xml",
                    },
                    timeout=30,
                )

                if resp.status_code != 200:
                    continue

                # Parse XML to extract abstracts
                root = ET.fromstring(resp.content)

                for article_elem in root.findall(".//PubmedArticle"):
                    # Get PMID
                    pmid_elem = article_elem.find(".//PMID")
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text

                    # Get Abstract
                    abstract_parts = []

                    # Handle structured abstracts (with labels like Background, Methods, etc.)
                    abstract_elem = article_elem.find(".//Abstract")
                    if abstract_elem is not None:
                        for text_elem in abstract_elem.findall("AbstractText"):
                            label = text_elem.get("Label", "")
                            text = self._get_element_text(text_elem)

                            if text:
                                if label:
                                    abstract_parts.append(f"**{label}:** {text}")
                                else:
                                    abstract_parts.append(text)

                    if abstract_parts:
                        abstracts[pmid] = "\n\n".join(abstract_parts)

                    # Also try to get keywords and MeSH headings
                    # (useful for the AI to understand the article)
                    mesh_headings = []
                    for mesh_elem in article_elem.findall(
                        ".//MeshHeading/DescriptorName"
                    ):
                        if mesh_elem.text:
                            mesh_headings.append(mesh_elem.text)

                    if mesh_headings and pmid in abstracts:
                        abstracts[
                            pmid
                        ] += f"\n\n**MeSH Keywords:** {', '.join(mesh_headings[:10])}"

                    keywords = []
                    for kw_elem in article_elem.findall(".//Keyword"):
                        if kw_elem.text:
                            keywords.append(kw_elem.text)

                    if keywords and pmid in abstracts:
                        abstracts[
                            pmid
                        ] += f"\n**Author Keywords:** {', '.join(keywords[:10])}"

            except ET.ParseError:
                # If XML parsing fails, try regex fallback
                self._fetch_abstracts_fallback(batch, abstracts, resp.text)
            except Exception:
                continue

        return abstracts

    def _get_element_text(self, elem) -> str:
        """
        Get all text from an XML element, including text in child elements.
        Handles cases like: <AbstractText>Some text <i>italic</i> more text</AbstractText>
        """
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            if child.text:
                parts.append(child.text)
            if child.tail:
                parts.append(child.tail)
        return " ".join(parts).strip()

    def _fetch_abstracts_fallback(
        self, pmids: List[str], abstracts: Dict, xml_text: str
    ):
        """Regex fallback if XML parsing fails"""
        try:
            for pmid in pmids:
                if pmid in abstracts:
                    continue

                # Find abstract text using regex
                pattern = (
                    rf"<PMID[^>]*>{re.escape(pmid)}</PMID>"
                    rf".*?"
                    rf"<Abstract>(.*?)</Abstract>"
                )
                match = re.search(pattern, xml_text, re.DOTALL)
                if match:
                    abstract_xml = match.group(1)
                    # Strip XML tags
                    text = re.sub(r"<[^>]+>", " ", abstract_xml)
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        abstracts[pmid] = text
        except Exception:
            pass

    # ================================================================
    # CORE: PUBMED QUERY ANALYSIS
    # ================================================================

    def _analyze_via_pubmed(self, query: str) -> Dict:
        """Let PubMed itself parse and understand the query"""

        cache_key = query.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {
            "original": query,
            "cleaned_query": query,
            "query_translation": "",
            "mesh_found": [],
            "result_count": 0,
        }

        try:
            resp = requests.get(
                f"{self.base_url}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": "0",
                    "usehistory": "n",
                },
                timeout=15,
            )

            if resp.status_code == 200:
                data = resp.json()
                esearch = data.get("esearchresult", {})
                result["result_count"] = int(esearch.get("count", 0))
                result["query_translation"] = esearch.get("querytranslation", "")

                if result["query_translation"]:
                    mesh = re.findall(
                        r'"([^"]+)"\[MeSH Terms\]', result["query_translation"]
                    )
                    result["mesh_found"] = list(dict.fromkeys(mesh))

            # Also check full phrase as MeSH
            resp2 = requests.get(
                f"{self.base_url}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": f'"{query}"[MeSH Terms]',
                    "retmode": "json",
                    "retmax": "0",
                },
                timeout=10,
            )

            if resp2.status_code == 200:
                data2 = resp2.json()
                trans2 = data2.get("esearchresult", {}).get("querytranslation", "")
                count2 = int(data2.get("esearchresult", {}).get("count", 0))
                if count2 > 0 and trans2:
                    extra = re.findall(r'"([^"]+)"\[MeSH Terms\]', trans2)
                    for t in extra:
                        if t not in result["mesh_found"]:
                            result["mesh_found"].append(t)

        except Exception:
            pass

        self._cache[cache_key] = result
        return result

    # ================================================================
    # QUERY TYPE DETECTION
    # ================================================================

    def _detect_query_type(self, query: str) -> str:
        q = query.lower()
        if any(
            w in q
            for w in [
                "guideline",
                "guidelines",
                "protocol",
                "recommendation",
                "consensus",
                "management of",
            ]
        ):
            return "guidelines"
        elif any(w in q for w in ["systematic review", "meta-analysis"]):
            return "systematic_review"
        elif any(w in q for w in ["outcome", "outcomes", "effectiveness", "efficacy"]):
            return "outcomes"
        elif any(w in q for w in ["prevalence", "incidence", "epidemiology"]):
            return "epidemiology"
        elif any(w in q for w in ["diagnosis", "diagnostic", "screening"]):
            return "diagnosis"
        elif any(w in q for w in ["treatment", "therapy", "drug", "medication"]):
            return "treatment"
        elif any(w in q for w in ["risk factor", "cause", "etiology"]):
            return "risk_factors"
        return "general"

    # ================================================================
    # SEARCH STRATEGIES
    # ================================================================

    def _iterative_search(self, query, analysis, query_type, max_results):
        strategies = self._build_strategies(query, analysis, query_type)
        return self._run_strategies(strategies, max_results)

    def _build_strategies(self, query, analysis, query_type):
        strategies = []
        mesh = analysis.get("mesh_found", [])
        count = analysis.get("result_count", 0)
        type_filter = self._get_type_filter(query_type)

        # Strategy 1: Original query (PubMed auto-maps it)
        if count > 0:
            strategies.append(("PubMed Auto-Mapping", query))

        # Strategy 2: MeSH + type filter
        if mesh and type_filter:
            mq = " AND ".join([f'"{t}"[MeSH]' for t in mesh[:4]])
            strategies.append((f"MeSH + {query_type} filter", f"{mq}{type_filter}"))

        # Strategy 3: MeSH only
        if mesh:
            mq = " AND ".join([f'"{t}"[MeSH]' for t in mesh[:4]])
            strategies.append(("MeSH Combined", mq))

        # Strategy 4: Core MeSH (top 2)
        if len(mesh) >= 2:
            strategies.append(("Core MeSH", f'"{mesh[0]}"[MeSH] AND "{mesh[1]}"[MeSH]'))

        # Strategy 5: Primary MeSH + filter
        if mesh and type_filter:
            strategies.append(
                (f"Primary MeSH + {query_type}", f'"{mesh[0]}"[MeSH]{type_filter}')
            )

        # Strategy 6: Title/Abstract
        strategies.append(("Title/Abstract", f"({query})[tiab]"))

        # Strategy 7: All fields
        strategies.append(("All Fields", query))

        return strategies

    def _pico_iterative_search(self, pico, pico_analysis, max_results):
        strategies = []

        # Build per-component queries
        comp_parts = []
        for comp, analysis in pico_analysis.items():
            mesh = analysis.get("mesh_found", [])
            if mesh:
                if len(mesh) > 1:
                    group = " OR ".join([f'"{t}"[MeSH]' for t in mesh[:3]])
                    comp_parts.append(f"({group})")
                else:
                    comp_parts.append(f'"{mesh[0]}"[MeSH]')
            else:
                comp_parts.append(f"({pico[comp]})")

        if len(comp_parts) >= 2:
            strategies.append(("Full PICO", " AND ".join(comp_parts)))

        combined = " ".join(pico.values())
        strategies.append(("Natural Language", combined))

        # P + I
        for pair_name, keys in [
            ("P+I", ["Population", "Intervention"]),
            ("P+O", ["Population", "Outcome"]),
        ]:
            parts = []
            for k in keys:
                if k in pico_analysis:
                    mesh = pico_analysis[k].get("mesh_found", [])
                    if mesh:
                        parts.append(f'"{mesh[0]}"[MeSH]')
                    elif k in pico:
                        parts.append(pico[k])
            if len(parts) == 2:
                strategies.append((pair_name, " AND ".join(parts)))

        strategies.append(("Broad", combined))

        return self._run_strategies(strategies, max_results)

    def _run_strategies(self, strategies, max_results):
        all_articles = []
        seen = set()
        log = []

        for name, sq in strategies:
            if not sq:
                continue
            results = self._run_search(sq, min(max_results * 2, 50))
            log.append({"name": name, "query": sq, "found": len(results)})

            for a in results:
                if a["pmid"] not in seen:
                    seen.add(a["pmid"])
                    a["found_via"] = name
                    all_articles.append(a)

            if len(all_articles) >= max_results * 3 and len(log) >= 3:
                break

        return all_articles, log

    def _get_type_filter(self, query_type):
        filters = {
            "guidelines": ' AND ("Practice Guideline"[PT] OR "Guideline"[PT] OR guideline[ti])',
            "systematic_review": ' AND ("Systematic Review"[PT] OR "Meta-Analysis"[PT])',
            "outcomes": ' AND ("Clinical Trial"[PT] OR "Comparative Study"[PT])',
            "diagnosis": " AND (diagnosis[ti] OR diagnostic[ti])",
            "epidemiology": " AND (prevalence[ti] OR incidence[ti] OR epidemiology[sh])",
            "treatment": ' AND ("Clinical Trial"[PT] OR "Randomized Controlled Trial"[PT])',
        }
        return filters.get(query_type, "")

    # ================================================================
    # RELEVANCE SCORING (now uses abstracts)
    # ================================================================

    def _score_relevance(self, articles, query, query_type):
        query_words = set(re.findall(r"[a-z]{3,}", query.lower()))
        stops = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "are",
            "was",
            "were",
            "been",
            "have",
            "has",
            "how",
            "what",
            "which",
            "current",
            "recent",
            "new",
            "using",
            "based",
        }
        query_words -= stops

        type_bonuses = {
            "guidelines": [
                "guideline",
                "guidelines",
                "recommendation",
                "consensus",
                "management",
            ],
            "systematic_review": ["systematic", "review", "meta-analysis"],
            "outcomes": ["outcome", "outcomes", "effectiveness", "efficacy"],
            "diagnosis": ["diagnosis", "diagnostic", "screening"],
            "treatment": ["treatment", "therapy", "therapeutic", "intervention"],
            "epidemiology": ["prevalence", "incidence", "epidemiology"],
        }

        for article in articles:
            score = 0
            title_lower = article.get("title", "").lower()
            abstract_lower = article.get("abstract", "").lower()

            # Title word overlap
            title_words = set(re.findall(r"[a-z]{3,}", title_lower))
            score += len(query_words & title_words) * 5

            # Abstract word overlap (worth less per word, but still valuable)
            if abstract_lower:
                abstract_words = set(re.findall(r"[a-z]{3,}", abstract_lower))
                score += min(15, len(query_words & abstract_words) * 2)

            # Query type bonus
            if query_type in type_bonuses:
                for w in type_bonuses[query_type]:
                    if w in title_lower:
                        score += 10
                    if w in abstract_lower:
                        score += 3

            # Recency
            year = self._extract_year(article.get("pubdate", ""))
            if year:
                if year >= 2023:
                    score += 8
                elif year >= 2020:
                    score += 5
                elif year >= 2015:
                    score += 2

            # Journal quality
            journal_lower = article.get("journal", "").lower()
            if any(
                j in journal_lower
                for j in [
                    "lancet",
                    "bmj",
                    "jama",
                    "new england",
                    "cochrane",
                    "pediatrics",
                    "annals",
                    "nature",
                    "plos",
                ]
            ):
                score += 5

            # Bonus if abstract exists (more informative article)
            if abstract_lower:
                score += 3

            article["relevance_score"] = score

        articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return articles

    # ================================================================
    # PUBMED SEARCH API
    # ================================================================

    def _run_search(self, query, max_results):
        max_results = self._safe_int(max_results, 10, 1, 50)
        try:
            resp = requests.get(
                f"{self.base_url}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": str(query),
                    "retmode": "json",
                    "retmax": str(max_results),
                    "sort": "relevance",
                },
                timeout=20,
            )
            resp.raise_for_status()
            esearch = resp.json().get("esearchresult", {})
            if "ERROR" in esearch:
                return []

            ids = esearch.get("idlist", esearch.get("IdList", []))
            if not ids:
                return []

            resp = requests.get(
                f"{self.base_url}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                timeout=20,
            )
            resp.raise_for_status()
            summaries = resp.json().get("result", {})

            articles = []
            for aid in ids:
                if aid not in summaries:
                    continue
                art = summaries[aid]
                if not isinstance(art, dict) or "title" not in art:
                    continue
                articles.append(
                    {
                        "title": art.get("title", "Untitled"),
                        "authors": ", ".join(
                            a["name"]
                            for a in art.get("authors", [])
                            if isinstance(a, dict) and a.get("name")
                        ),
                        "pubdate": art.get("pubdate", ""),
                        "journal": art.get("fulljournalname", ""),
                        "volume": art.get("volume", ""),
                        "issue": art.get("issue", ""),
                        "pages": art.get("pages", ""),
                        "doi": next(
                            (
                                x["value"]
                                for x in art.get("articleids", [])
                                if isinstance(x, dict) and x.get("idtype") == "doi"
                            ),
                            "",
                        ),
                        "pmid": aid,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{aid}/",
                        "abstract": "",  # Will be filled by _fetch_abstracts
                    }
                )
            return articles
        except Exception:
            return []

    # ================================================================
    # FORMATTING
    # ================================================================

    def _format_results(
        self, query, analysis, query_type, search_log, articles, total, show_abstracts
    ):
        md = "# 📚 PubMed Search Results\n\n"
        md += f"**Your Question:** {query}\n\n"

        # PubMed understanding
        md += "## 🧠 How PubMed Understood Your Query\n\n"
        md += f"**Query Type:** {query_type.replace('_', ' ').title()}\n\n"

        if analysis["mesh_found"]:
            md += "**MeSH Terms:**\n"
            for t in analysis["mesh_found"]:
                md += f"- `{t}`\n"
            md += "\n"

        if analysis["query_translation"]:
            md += f"**PubMed Translation:**\n```\n{analysis['query_translation']}\n```\n\n"

        # Search process
        md += "## 🔧 Search Process\n\n"
        md += f"Ran {len(search_log)} strategies, collected {total} candidates, showing top {len(articles)}.\n\n"
        for s in search_log:
            icon = "✅" if s["found"] > 0 else "⭕"
            md += f"{icon} **{s['name']}** → {s['found']}  \n"
        md += "\n"

        # Results
        md += f"## 📄 Top {len(articles)} Results\n\n"

        # Count how many have abstracts
        with_abstracts = sum(1 for a in articles if a.get("abstract"))
        if with_abstracts > 0:
            md += f"*📋 {with_abstracts}/{len(articles)} articles have abstracts available for AI analysis*\n\n"

        md += self._format_article_list(articles, show_abstracts=show_abstracts)

        # Synthesis hint
        if with_abstracts > 0:
            md += "## 🤖 AI Can Now Analyze These\n\n"
            md += "The abstracts have been loaded. You can now ask me:\n\n"
            md += f"> Based on these results, what are the key findings about {query}?\n\n"
            md += f"> Summarize the evidence on {query}\n\n"
            md += f"> What do these studies conclude about {query}?\n\n"

        md += self._format_next_steps()
        return md

    def _format_article_list(self, articles, show_abstracts=True):
        md = ""
        for i, a in enumerate(articles):
            score = a.get("relevance_score", 0)
            stars = min(5, max(1, score // 5))

            md += f"### {i+1}. {a.get('title', 'Untitled')}\n\n"

            if a.get("authors"):
                auths = a["authors"].split(", ")
                if len(auths) > 3:
                    md += f"**Authors:** {', '.join(auths[:3])}, et al.\n\n"
                else:
                    md += f"**Authors:** {a['authors']}\n\n"

            info = []
            if a.get("journal"):
                info.append(f"*{a['journal']}*")
            if a.get("pubdate"):
                info.append(a["pubdate"])
            v = a.get("volume", "")
            if v:
                if a.get("issue"):
                    v += f"({a['issue']})"
                if a.get("pages"):
                    v += f":{a['pages']}"
                info.append(v)
            if info:
                md += f"**Published:** {' | '.join(info)}\n\n"

            if a.get("doi"):
                md += f"**DOI:** [{a['doi']}](https://doi.org/{a['doi']})\n\n"

            md += f"🔗 [PubMed PMID {a['pmid']}]({a['url']}) · {'⭐' * stars}\n\n"

            # ABSTRACT — the key addition
            if show_abstracts and a.get("abstract"):
                abstract = a["abstract"]
                md += f"<details>\n<summary>📋 Abstract (click to expand)</summary>\n\n"
                md += f"{abstract}\n\n"
                md += f"</details>\n\n"

            md += "---\n\n"

        return md

    def _format_next_steps(self):
        md = "\n## 💡 What You Can Do Next\n\n"
        md += "| Command | What You Get |\n"
        md += "|---------|-------------|\n"
        md += "| `get results as list` | Numbered reference list |\n"
        md += "| `get results as ris` | RIS file for Zotero import |\n"
        md += "| `get results as summary` | AI synthesis of findings |\n"
        md += "| `get results as abstracts` | All abstracts for reading |\n"
        md += "| `get results as detailed` | Full metadata for every article |\n"
        md += "\n"
        return md

    def _format_no_results(self, query, analysis, search_log):
        md = f"# No Results Found\n\n**Query:** {query}\n\n"
        if analysis["query_translation"]:
            md += f"**PubMed interpreted as:**\n```\n{analysis['query_translation']}\n```\n\n"
        for s in search_log:
            md += f"❌ {s['name']}: `{s['query']}`\n\n"
        md += "Try simpler phrasing or use `find_mesh` to check terms.\n"
        return md

    # ================================================================
    # OUTPUT FORMATS
    # ================================================================

    def _format_reference_list(self):
        md = f"# 📋 References ({len(self._last_results)})\n\n"
        for i, a in enumerate(self._last_results):
            authors = a.get("authors", "")
            auths = authors.split(", ")
            if len(auths) > 3:
                authors = ", ".join(auths[:3]) + ", et al."
            year = self._extract_year(a.get("pubdate", ""))
            year_str = f" ({year})" if year else ""
            title = a.get("title", "").rstrip(".")
            ref = f"{i+1}. {authors}.{year_str} {title}."
            if a.get("journal"):
                ref += f" *{a['journal']}*."
            v = a.get("volume", "")
            if v:
                if a.get("issue"):
                    v += f"({a['issue']})"
                if a.get("pages"):
                    v += f":{a['pages']}"
                ref += f" {v}."
            if a.get("doi"):
                ref += f" doi:{a['doi']}"
            ref += f" PMID:{a['pmid']}"
            md += f"{ref}\n\n"
        md += "> `get results as ris` for Zotero export\n"
        return md

    def _export_ris(self):
        ris = ""
        for a in self._last_results:
            ris += self._to_ris(a)
        return (
            f"# 📥 RIS Export ({len(self._last_results)} refs)\n\n"
            "1. Copy code block → 2. Save as `.ris` → 3. Zotero Import\n\n"
            f"```ris\n{ris}```\n"
        )

    def _format_abstracts_only(self):
        """Output all abstracts — perfect for AI to read and synthesize"""
        md = f"# 📋 Abstracts ({len(self._last_results)} articles)\n\n"
        md += f"**Search:** {self._last_query}\n\n---\n\n"

        for i, a in enumerate(self._last_results):
            year = self._extract_year(a.get("pubdate", "")) or "n.d."
            auths = a.get("authors", "").split(", ")
            first = auths[0] if auths else "Unknown"

            md += f"## {i+1}. {a.get('title', 'Untitled')}\n"
            md += f"*{first} et al. ({year}) — {a.get('journal', '')}*\n\n"

            if a.get("abstract"):
                md += f"{a['abstract']}\n\n"
            else:
                md += "*No abstract available.*\n\n"

            if a.get("doi"):
                md += f"DOI: {a['doi']}\n"
            md += f"PMID: {a['pmid']}\n\n"
            md += "---\n\n"

        return md

    def _synthesize(self):
        """Synthesis using abstract content"""
        articles = self._last_results
        md = f"# 📊 Research Summary\n\n"
        md += f"**Question:** {self._last_query}\n"
        md += f"**Articles:** {len(articles)}\n\n"

        years = [self._extract_year(a.get("pubdate", "")) for a in articles]
        years = [y for y in years if y]
        if years:
            md += f"**Publication Range:** {min(years)}–{max(years)}\n\n"

        with_abstracts = sum(1 for a in articles if a.get("abstract"))
        md += f"**Abstracts Available:** {with_abstracts}/{len(articles)}\n\n"

        # Journals
        journals = {}
        for a in articles:
            j = a.get("journal", "Unknown")
            journals[j] = journals.get(j, 0) + 1

        md += "## Sources\n\n"
        for j, c in sorted(journals.items(), key=lambda x: -x[1])[:8]:
            md += f"- {j} ({c})\n"
        md += "\n"

        # Key themes from titles AND abstracts
        all_text = ""
        for a in articles:
            all_text += " " + a.get("title", "")
            all_text += " " + a.get("abstract", "")

        word_freq = {}
        stops = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "was",
            "were",
            "been",
            "have",
            "has",
            "study",
            "review",
            "analysis",
            "patients",
            "results",
            "methods",
            "conclusion",
            "background",
            "objective",
            "purpose",
            "clinical",
            "using",
            "based",
            "among",
            "between",
            "however",
            "conclusion",
            "findings",
            "group",
            "compared",
            "significantly",
            "associated",
            "included",
            "data",
        }

        for w in re.findall(r"[a-z]{4,}", all_text.lower()):
            if w not in stops:
                word_freq[w] = word_freq.get(w, 0) + 1

        md += "## Key Themes\n\n"
        for w, c in sorted(word_freq.items(), key=lambda x: -x[1])[:15]:
            if c >= 3:
                md += f"- **{w}** (appears {c} times)\n"
        md += "\n"

        # Article summaries with abstract snippets
        md += "## Article Summaries\n\n"
        for i, a in enumerate(articles[:15]):
            year = self._extract_year(a.get("pubdate", "")) or "n.d."
            auths = a.get("authors", "").split(", ")
            first = auths[0] if auths else "Unknown"

            md += f"### {i+1}. {first} ({year})\n"
            md += f"**{a.get('title', '')}**\n"
            md += f"*{a.get('journal', '')}*\n\n"

            if a.get("abstract"):
                # Show first 300 chars of abstract
                snippet = a["abstract"][:300]
                if len(a["abstract"]) > 300:
                    snippet += "..."
                md += f"{snippet}\n\n"

        md += "---\n"
        md += "*Full abstracts available — ask me to analyze specific findings.*\n"
        return md

    def _format_detailed(self):
        md = f"# 📑 Detailed ({len(self._last_results)})\n\n"
        for i, a in enumerate(self._last_results):
            md += f"## {i+1}. {a.get('title', 'Untitled')}\n\n"
            md += f"- **Authors:** {a.get('authors', 'Unknown')}\n"
            md += f"- **Journal:** {a.get('journal', 'Unknown')}\n"
            md += f"- **Date:** {a.get('pubdate', 'Unknown')}\n"
            if a.get("doi"):
                md += f"- **DOI:** [{a['doi']}](https://doi.org/{a['doi']})\n"
            md += f"- **PMID:** [{a['pmid']}]({a['url']})\n"
            md += f"- **Relevance:** {a.get('relevance_score', 0)} · via {a.get('found_via', '?')}\n"
            if a.get("abstract"):
                md += f"\n**Abstract:**\n\n{a['abstract']}\n"
            md += "\n---\n\n"
        return md

    # ================================================================
    # UTILITIES
    # ================================================================

    def _to_ris(self, article):
        ris = "TY  - JOUR\n"
        if article.get("authors"):
            for a in article["authors"].split(", "):
                if a.strip():
                    ris += f"AU  - {a.strip()}\n"
        ris += f"T1  - {article.get('title', '').rstrip('.')}\n"
        if article.get("journal"):
            ris += f"JO  - {article['journal']}\n"
        if article.get("pubdate"):
            m = re.search(r"(\d{4})", article["pubdate"])
            if m:
                ris += f"PY  - {m.group(1)}\n"
            ris += f"DA  - {article['pubdate']}\n"
        if article.get("volume"):
            ris += f"VL  - {article['volume']}\n"
        if article.get("issue"):
            ris += f"IS  - {article['issue']}\n"
        if article.get("pages"):
            if "-" in article["pages"]:
                sp, ep = article["pages"].split("-", 1)
                ris += f"SP  - {sp.strip()}\n"
                ris += f"EP  - {ep.strip()}\n"
            else:
                ris += f"SP  - {article['pages']}\n"
        if article.get("doi"):
            ris += f"DO  - {article['doi']}\n"
        if article.get("url"):
            ris += f"UR  - {article['url']}\n"
        if article.get("abstract"):
            ris += f"AB  - {article['abstract'][:2000]}\n"
        ris += "ER  -\n\n"
        return ris

    def _extract_year(self, pubdate):
        if not pubdate:
            return None
        m = re.search(r"(\d{4})", str(pubdate))
        return int(m.group(1)) if m else None

    def _safe_int(self, value, default=10, minimum=1, maximum=50):
        try:
            r = int(float(str(value)))
        except (TypeError, ValueError):
            r = default
        return max(minimum, min(maximum, r))
