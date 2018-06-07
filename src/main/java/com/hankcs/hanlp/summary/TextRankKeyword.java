package com.hankcs.hanlp.summary;

import com.hankcs.hanlp.algorithm.MaxHeap;
import com.hankcs.hanlp.seg.common.Term;

import java.util.*;

/**
 * 基于TextRank算法的关键字提取，适用于单文档
 * <p>
 * 参考论文:TextRank: Bringing Order into Texts by Rada Mihalcea
 *
 * @author hankcs
 */
public class TextRankKeyword extends KeywordExtractor {
    /**
     * 提取多少个关键字
     */
    int nKeyword = 10;
    /**
     * 阻尼系数（ＤａｍｐｉｎｇＦａｃｔｏｒ），一般取值为0.85
     */
    final static float d = 0.85f;//论文中引用的值
    /**
     * 最大迭代次数
     */
    public static int max_iter = 200;
    final static float min_diff = 0.001f;

    /**
     * 提取关键词
     *
     * @param document 文档内容
     * @param size     希望提取几个关键词
     * @return 一个列表
     */
    public static List<String> getKeywordList(String document, int size) {
        TextRankKeyword textRankKeyword = new TextRankKeyword();
        textRankKeyword.nKeyword = size;

        return textRankKeyword.getKeyword(document);
    }

    /**
     * 提取关键词
     *
     * @param content
     * @return
     */
    public List<String> getKeyword(String content) {
        Set<Map.Entry<String, Float>> entrySet = getTermAndRank(content, nKeyword).entrySet();
        List<String> result = new ArrayList<String>(entrySet.size());
        for (Map.Entry<String, Float> entry : entrySet) {
            result.add(entry.getKey());
        }
        return result;
    }

    /**
     * 返回全部分词结果和对应的rank
     *
     * @param content
     * @return
     */
    public Map<String, Float> getTermAndRank(String content) {
        assert content != null;
        List<Term> termList = defaultSegment.seg(content);
        return getRank(termList);
    }

    /**
     * 返回分数最高的前size个分词结果和对应的rank
     *
     * @param content
     * @param size
     * @return
     */
    public Map<String, Float> getTermAndRank(String content, Integer size) {
        Map<String, Float> map = getTermAndRank(content);
        Map<String, Float> result = new LinkedHashMap<String, Float>();
        for (Map.Entry<String, Float> entry : new MaxHeap<Map.Entry<String, Float>>(size,
            new Comparator<Map.Entry<String, Float>>() {
                @Override
                public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                    return o1.getValue().compareTo(o2.getValue());
                }
            }).addAll(map.entrySet()).toList()) {
            result.put(entry.getKey(), entry.getValue());
        }

        return result;
    }

    /**
     * 使用已经分好的词来计算rank
     *
     * @param termList
     * @return
     */
    public Map<String, Float> getRank(List<Term> termList) {
        List<String> wordList = new ArrayList<String>(termList.size());//wordList 可能包含重复的 term
        for (Term t : termList) {
            if (shouldInclude(t)) {
                wordList.add(t.word);
            }
        }
        //        System.out.println(wordList);
        Map<String, Set<String>> words = new TreeMap<String, Set<String>>();
        Queue<String> que = new LinkedList<String>();
        for (String w : wordList) {
            if (!words.containsKey(w)) {
                //排除了 wordList 中的重复term, 对每个已去重的term, 用 TreeSet<String> 保存该term的邻接顶点
                words.put(w, new TreeSet<String>());
            }
            // 复杂度O(n-1)
            if (que.size() >= 5) {
                //窗口的大小为5,是写死的. 对于一个term_A而言, 它的前4个term、后4个term 都属于term_A的邻接点
                que.poll();
            }
            for (String qWord : que) {
                if (w.equals(qWord)) {
                    continue;
                }
                //既然是邻居,那么关系是相互的,遍历一遍即可
                words.get(w).add(qWord);
                words.get(qWord).add(w);
            }
            que.offer(w);
        }
        //        System.out.println(words);
        Map<String, Float> score = new HashMap<String, Float>();//保存最终每个关键词的得分
        //依据TF来设置初值,  words 代表的是 一张 无向图
        for (Map.Entry<String, Set<String>> entry : words.entrySet()) {
            score.put(entry.getKey(), sigMoid(entry.getValue().size()));//无向图的每个顶点 得分值 初始化
        }
        for (int i = 0; i < max_iter; ++i) {
            Map<String, Float> m = new HashMap<String, Float>();//保存每一轮迭代中每个关键词的得分
            float max_diff = 0;

            //对每一个顶点 WS(V_i)
            for (Map.Entry<String, Set<String>> entry : words.entrySet()) {
                String key = entry.getKey();
                Set<String> value = entry.getValue();//获取所有投票给key的顶点
                m.put(key, 1 - d);//先把公式 WS(V_i)的 前半部分 1-d 保存起来

                //基于顶点V_i 的所有邻接点 V_j 计算得分值
                for (String element : value) {
                    //element顶点一共给多少顶点投了票, size相当于 \Sigma_{V_k belongs to Out(V_j) w_{jk}}
                    int size = words.get(element).size();
                    if (key.equals(element) || size == 0)
                        continue;

                    /**m.get(key) 取出 1-d,
                     *score.get(element) == null ? 0 : score.get(element), 上面使用了Sigmod初始化,所以这里不太可能为null
                     *
                     * 先 m.get(key) 取出上一次迭代的结果, 然后再 加上 d/size*score.get(element)) 相当于当前 element的投票权重
                     * 然后再 for 循环,就把 对所有指向 key 的顶点集合value(Set<String>value)累加.
                     * for 循环相当于 \Sigma_{v_j belongs to In(V_i)}
                     */
                    m.put(key, m.get(key) + d / size * (score.get(element) == null ? 0 : score.get(element)));
                }
                max_diff = Math.max(max_diff, Math.abs(m.get(key) - (score.get(key) == null ? 0 : score.get(key))));
            }
            score = m;

            //已经达到了迭代阈值,不需要在进行下一轮的迭代了
            if (max_diff <= min_diff)
                break;
        }

        return score;
    }

    /**
     * sigmoid函数
     *
     * @param value
     * @return
     */
    public static float sigMoid(float value) {
        return (float) (1d / (1d + Math.exp(-value)));
    }
}
