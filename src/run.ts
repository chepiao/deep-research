import { spawn } from 'child_process';
import * as fsSync from 'fs';
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import * as readline from 'readline';

import { getModel } from './ai/providers';
import { generateFeedback } from './feedback';

function log(...args: any[]) {
  console.log(...args);
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuestion(query: string): Promise<string> {
  return new Promise(resolve => {
    rl.question(query, answer => {
      resolve(answer);
    });
  });
}

function formatDateTime(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function validateName(name: string): boolean {
  return /^[\w\u4e00-\u9fff-]+$/.test(name);
}

// 后台模式：从配置文件读取参数，执行调研，写报告
async function runBackground(configPath: string) {
  const configRaw = await fs.readFile(configPath, 'utf-8');
  const config = JSON.parse(configRaw);

  // 清理临时配置文件
  await fs.unlink(configPath).catch(() => {});

  // 重定向 stdout/stderr 到日志文件
  const logStream = fsSync.createWriteStream(config.logFile, { flags: 'a' });

  const origStdoutWrite = process.stdout.write.bind(process.stdout);
  const origStderrWrite = process.stderr.write.bind(process.stderr);

  process.stdout.write = function (chunk: any, ...args: any[]) {
    logStream.write(chunk);
    return true;
  };
  process.stderr.write = function (chunk: any, ...args: any[]) {
    logStream.write(chunk);
    return true;
  };

  // 动态导入，避免顶层加载 deep-research 的副作用
  const { deepResearch, writeFinalAnswer, writeFinalReport } = await import(
    './deep-research'
  );

  log(`[${new Date().toISOString()}] Background research started`);
  log(`Report file: ${config.reportFile}`);
  log(`Log file: ${config.logFile}\n`);

  try {
    const { learnings, visitedUrls } = await deepResearch({
      query: config.combinedQuery,
      breadth: config.breadth,
      depth: config.depth,
    });

    log(`\n\nLearnings:\n\n${learnings.join('\n')}`);
    log(
      `\n\nVisited URLs (${visitedUrls.length}):\n\n${visitedUrls.join('\n')}`,
    );

    if (config.isReport) {
      log('Writing final report...');
      const report = await writeFinalReport({
        prompt: config.combinedQuery,
        learnings,
        visitedUrls,
      });
      log('Report generation complete, writing to file...');
      await fs.writeFile(config.reportFile, report, 'utf-8');
      log(`Report saved to ${config.reportFile}`);
    } else {
      const answer = await writeFinalAnswer({
        prompt: config.combinedQuery,
        learnings,
      });
      await fs.writeFile(config.reportFile, answer, 'utf-8');
      log(`Answer saved to ${config.reportFile}`);
    }

    log(`\n[${new Date().toISOString()}] Research completed successfully`);
  } catch (err) {
    log(`\n[${new Date().toISOString()}] Research failed: ${err}`);
    throw err;
  } finally {
    logStream.end();
  }
}

// 交互模式：收集参数后 fork 后台进程
async function runInteractive() {
  const args = process.argv.slice(2);
  let taskName = '';

  if (args.length > 0) {
    taskName = args[0];
    if (!validateName(taskName)) {
      console.error(
        `Error: Invalid task name "${taskName}". Only letters, numbers, Chinese characters, underscores, and hyphens are allowed.`,
      );
      process.exit(1);
    }
  }

  console.log('Using model: ', getModel().modelId);

  const initialQuery = await askQuestion('What would you like to research? ');

  if (!taskName) {
    taskName = initialQuery.slice(0, 20).replace(/[^\w\u4e00-\u9fff-]/g, '_');
  }

  const breadth =
    parseInt(
      await askQuestion(
        'Enter research breadth (recommended 2-10, default 4): ',
      ),
      10,
    ) || 4;
  const depth =
    parseInt(
      await askQuestion('Enter research depth (recommended 1-5, default 2): '),
      10,
    ) || 2;
  const isReport =
    (await askQuestion(
      'Do you want to generate a long report or a specific answer? (report/answer, default report): ',
    )) !== 'answer';

  let combinedQuery = initialQuery;
  if (isReport) {
    log(`Creating research plan...`);

    const followUpQuestions = await generateFeedback({
      query: initialQuery,
    });

    log(
      '\nTo better understand your research needs, please answer these follow-up questions:',
    );

    const answers: string[] = [];
    for (const question of followUpQuestions) {
      const answer = await askQuestion(`\n${question}\nYour answer: `);
      answers.push(answer);
    }

    combinedQuery = `
Initial Query: ${initialQuery}
Follow-up Questions and Answers:
${followUpQuestions.map((q: string, i: number) => `Q: ${q}\nA: ${answers[i]}`).join('\n')}
`;
  }

  rl.close();

  const now = new Date();
  const ts = formatDateTime(now);
  const prefix = isReport ? 'report' : 'answer';
  const reportFile = `${prefix}_${taskName}_${ts}.md`;
  const logFile = `report_${taskName}_${ts}.log`;

  const config = {
    combinedQuery,
    breadth,
    depth,
    isReport,
    reportFile,
    logFile,
  };
  const configPath = path.join(
    os.tmpdir(),
    `deep-research-${ts}-${process.pid}.json`,
  );
  await fs.writeFile(configPath, JSON.stringify(config), 'utf-8');

  // 使用 npx tsx 启动后台进程，确保 .env.local 被加载
  const child = spawn(
    'npx',
    ['tsx', '--env-file=.env.local', path.resolve(__dirname, 'run.ts'), '--background', configPath],
    {
      detached: true,
      stdio: 'ignore',
      cwd: process.cwd(),
      env: { ...process.env },
    },
  );
  child.unref();

  console.log(`\nResearch moved to background.`);
  console.log(`  Log:     ${logFile}`);
  console.log(`  Report:  ${reportFile}`);
  console.log(`\nYou can start a new research task now.`);
}

// 入口：根据参数决定运行模式
const bgIdx = process.argv.indexOf('--background');
if (bgIdx !== -1) {
  const configPath = process.argv[bgIdx + 1];
  runBackground(configPath).catch(err => {
    console.error(`Background research failed: ${err}`);
    process.exit(1);
  });
} else {
  runInteractive().catch(console.error);
}
